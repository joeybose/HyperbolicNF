import os
import wandb
import torch
import numpy as np
import copy
from torch import optim
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from visualization.utils import draw_nx_graph, draw_pyg_graph
import sys
import networkx as nx
import ipdb

sys.path.append("..")  # Adds higher directory to python modules path.
from flows.flows import MAFRealNVP, RealNVP, FlowSequential, AllTangentRealNVP, TangentRealNVP
from flows.flows import WrappedRealNVP
from distributions.normal import EuclideanNormal
from distributions.wrapped_normal import HyperboloidWrappedNormal
from utils.hyperbolics import exp_map_mu0, inverse_exp_map_mu0
from utils.eval_utils import evaluate_generated
from utils.utils import perm_node_feats, create_selfloop_edges

kwargs_flows = {'MAFRealNVP': MAFRealNVP, 'RealNVP': RealNVP,
                'AllTangentRealNVP': AllTangentRealNVP, 'TangentRealNVP':
                    TangentRealNVP, 'WrappedRealNVP': WrappedRealNVP}


def ll_estimate(model, x):
    if model.ll_estimate == 'mc':
        return model.MC_log_likelihood(x)
    elif model.ll_estimate == 'iwae':
        return model.iwae(x)
    else:
        print('Set the log likelihood estimate as either mc (Monte Carlo) or iwae (importance weighted autoencoder).')
        return 0, None


def train_wrapper(args, opt, train_loader, test_loader, model):
    if args.dataset in ["mnist", "bdp", "pbt"]:
        train_metric = train_feedforward(args, opt, train_loader, test_loader, model)
    elif args.dataset in ["cora", "pubmed", "ppi", "disease_lp", "csphd",
                          "phylo", "diseases", "wordnet-noun", "wordnet-mammal"]:
        train_metric = train_graph(args, opt, train_loader, test_loader, model)
    elif args.dataset in ["lobster", "grid", "prufer"]:
        train_metric = train_generation_wrapper(args, opt, train_loader, test_loader, model)
    else:
        raise ValueError(f"Unknown dataset type: '{args.dataset}'.")
    return train_metric


def train_feedforward(args, opt, train_loader, test_loader, model):
    ll_avg, train_loss_avg, recon_loss_avg, kl_avg = [], [], [], []
    num_batches, test_ll = 0, 0
    preclamp_cond = (args.flow_model and args.model.lower() != 'euclidean')
    for epoch in range(0, args.epochs):
        train_loss_avg.append(0)
        recon_loss_avg.append(0)
        ll_avg.append(0)
        kl_avg.append(0)
        for data, target in train_loader:
            if model.type != 'euclidean' and epoch < 10:
                model.radius.data = torch.ones_like(model.radius.data).to(torch.float64) * (11 - epoch)
            x, target = data.to(args.dev), target.to(args.dev)
            opt.zero_grad()
            x_tilde, kld = model(x)
            assert x.shape == x_tilde.shape
            ### Reconstruction loss ###
            recon_loss = model.recon_loss(x_tilde, x).sum(dim=-1)
            ### Negative Elbo ##
            loss = (recon_loss + model.beta * kld).sum(dim=0)

            # Estimate Log-Liklihood on Training data
            # TODO: Make ll consistent -- either mc and iwae positive or negative
            with torch.no_grad():
                ll, mi = ll_estimate(model, x)
            ll_avg[-1] += ll.sum(dim=-1).item()
            train_loss_avg[-1] += -1 * loss.item()
            recon_loss_avg[-1] += recon_loss.sum(dim=-1).item()
            kl_avg[-1] += kld.sum(dim=-1).item()
            assert torch.isfinite(loss).all()
            loss.backward()
            opt.step()
            num_batches += 1

        ### Compute Epoch Stats
        train_loss_avg[-1] /= len(train_loader.dataset)
        recon_loss_avg[-1] /= len(train_loader.dataset)
        kl_avg[-1] /= len(train_loader.dataset)
        ll_avg[-1] /= len(train_loader.dataset)
        print("Epoch [%d/ %d]: Radius: %f | Train LL: %f | Elbo: %f | Reconstruction: %f | Pseudo KL: %f"
              % (epoch + 1, args.epochs, model.radius.data, ll_avg[-1], train_loss_avg[-1], recon_loss_avg[-1],
                 kl_avg[-1]))

        if train_loss_avg[-1] > 0:
            raise ValueError('Elbo > 0, something went wrong')

        if epoch % args.log_freq == 0:
            test_ll = test_feedforward(test_loader, model, epoch, args)

        ### Online Logging
        train_ll_metric = 'Train MC Log Likelihood'
        test_ll_metric = 'Test MC Log Likelihood'
        train_loss_metric = 'Elbo'
        recon_loss_metric = 'Reconstruction'
        kl_metric = 'Pseudo KL-Divergence'
        radius = 'Radius'
        preclamp_norm = model.flow_model.preclamp_norm.item() if preclamp_cond else 0
        if args.wandb:
            wandb.log({train_ll_metric: ll_avg[-1], test_ll_metric: test_ll, \
                       train_loss_metric: train_loss_avg[-1], \
                       recon_loss_metric: recon_loss_avg[-1], \
                       kl_metric: kl_avg[-1], "x": epoch, \
                       radius: model.radius.data.item(), \
                       "Pre-Clamp Norm": preclamp_norm})

    # Final Test
    test_ll_avg = test_feedforward(test_loader, model, epoch, args)
    return ll_avg[-1], model


def test_feedforward(test_loader, model, epoch, args):
    ll_avg, num_batches = 0, 0
    for data, target in test_loader:
        x, target = data.to(args.dev), target.to(args.dev)
        # TODO: Make ll consistent -- either mc and iwae positive or negative
        ll, mi = ll_estimate(model, x)
        ll_avg = ll.sum().item() / len(x)
        num_batches += 1
        print("Test Epoch [%d/ %d]: Radius: %f | LL: %f" % (epoch + 1, args.epochs, model.radius.data, ll_avg))
        ll_metric = 'Test MC Log Likelihood'
        if args.wandb:
            wandb.log({ll_metric: ll_avg})
        if num_batches > 0:
            break

    return ll_avg


def get_eval_set(args, data):
    if args.eval_set == 'validation':
        eval_pos_edge_index = data.val_pos_edge_index
        eval_neg_edge_index = data.val_neg_edge_index
    else:
        eval_pos_edge_index = data.test_pos_edge_index
        eval_neg_edge_index = data.test_neg_edge_index

    return eval_pos_edge_index.to(args.dev), eval_neg_edge_index.to(args.dev)


def train_graph(args, opt, train_loader, test_loader, model):
    model.train()
    preclamp_cond = (args.flow_model and args.model.lower() != 'euclidean')
    for i, data in enumerate(train_loader):
        print("### Graph %d ###" % (i))
        data = model.split_edges(data)
        x, train_pos_edge_index = data.x.to(args.dev), data.train_pos_edge_index.to(args.dev)
        for epoch in range(0, args.epochs):
            opt.zero_grad()
            # Randomly Permute Random Node Feats
            if args.use_rand_feats:
                x = perm_node_feats(x)

            if model.type != 'euclidean' and epoch < 10:
                model.encoder.radius.data = torch.ones_like(model.encoder.radius.data).to(torch.float64) * (11 - epoch)
            z, z_k = model.encode(x, train_pos_edge_index)
            recon_loss = model.recon_loss(z_k, train_pos_edge_index)
            kl_loss = (1 / data.num_nodes) * model.kl_loss(z, z_k)
            train_loss = recon_loss + args.beta*kl_loss
            train_loss.backward()
            opt.step()
            eval_pos_edge_index, eval_neg_edge_index = get_eval_set(args, data)
            auc, ap = test_graph(model, x, train_pos_edge_index, \
                                 eval_pos_edge_index, eval_neg_edge_index)
            mrr, hits1, hits3, hits10 = model.ranking_metrics(z_k, eval_pos_edge_index)
            print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, Pseudo KL:{:.4f},Elbo: {:.4f}, MRR: {:.4f}, Hits1: {:.2f}, Hits3: {:.2f}, Hits10: {:.2f}'.format(epoch, auc, ap, kl_loss.item(), -1*train_loss.item(), mrr, hits1, hits3, hits10))

            ### Online Logging
            test_auc_metric = 'Test AUC'
            test_ap_metric = 'Test AP'
            train_loss_metric = 'Train Loss'
            elbo_metric = 'Elbo'
            recon_loss_metric = 'Reconstruction'
            kl_metric = 'Pseudo KL-Divergence'
            radius = 'Radius'
            preclamp_norm = model.encoder.flow_model.preclamp_norm.item() if preclamp_cond else 0
            if args.wandb:
                wandb.log({test_auc_metric: auc, test_ap_metric: ap,
                           train_loss_metric: train_loss.item(),
                           elbo_metric: -1 * train_loss.item(),
                           recon_loss_metric: recon_loss.item(),
                           kl_metric: kl_loss.item(), "x": epoch,
                           radius: model.encoder.radius.data.item(),
                           "MRR": mrr,
                           "Hits1": hits1,
                           "Hits3": hits3,
                           "Hits10": hits10,
                           "Pre-Clamp Norm": preclamp_norm,
                           'dataset': args.dataset,
                           'epochs': args.epochs,
                           'eval_set': args.eval_set,
                           'model': args.model,
                           'flow_model': args.flow_model,
                           'n_blocks': args.n_blocks,
                           'z_dim': args.z_dim,
                           'hidden_dim': args.hidden_dim,
                           'flow_hidden_size': args.flow_hidden_size,
                           'fixed_curvature': args.fixed_curvature,
                           'decoder': args.decoder,
                           'temperature': args.temperature,
                           'decoder': args.decoder
                           })
    return auc, model

def train_graph_generation(args, opt, train_loader, test_loader, model):
    model.train()
    preclamp_cond = (args.flow_model and args.model.lower() != 'euclidean')
    for epoch in range(0, args.epochs):
        train_loss_avg, recon_loss_avg, kl_avg = [], [], []
        train_loss_avg.append(0)
        recon_loss_avg.append(0)
        kl_avg.append(0)
        for i, data_batch in enumerate(train_loader):
            opt.zero_grad()
            for idx, data in enumerate(data_batch):
                x, edge_index = data.x.to(args.dev), data.edge_index.to(args.dev)
                # Randomly Permute Random Node Feats
                if args.use_rand_feats:
                    x = perm_node_feats(x)

                if model.type != 'euclidean' and epoch < 10:
                    model.encoder.radius.data = torch.ones_like(model.encoder.radius.data).to(torch.float64) * (11 - epoch)
                z, z_k = model.encode(x, edge_index)
                recon_loss = model.recon_loss(z_k, edge_index)
                kl_loss = (1 / data.num_nodes) * model.kl_loss(z, z_k)
                train_loss = recon_loss + args.beta*kl_loss
                train_loss.backward()

                train_loss_avg[-1] += -1 * train_loss.item()
                recon_loss_avg[-1] += recon_loss.sum(dim=-1).item()
                kl_avg[-1] += kl_loss.item()
                ### Online Logging
                train_loss_metric = 'Train Loss'
                elbo_metric = 'Elbo'
                recon_loss_metric = 'Reconstruction'
                kl_metric = 'Pseudo KL-Divergence'
                radius = 'Radius'
                preclamp_norm = model.encoder.flow_model.preclamp_norm.item() if preclamp_cond else 0
                if args.wandb:
                    wandb.log({train_loss_metric: train_loss.item(),
                               elbo_metric: -1 * train_loss.item(),
                               recon_loss_metric: recon_loss.item(),
                               kl_metric: kl_loss.item(), "x": epoch,
                               radius: model.encoder.radius.data.item(),
                               "Pre-Clamp Norm": preclamp_norm,
                               'dataset': args.dataset,
                               'epochs': args.epochs,
                               'model': args.model,
                               'flow_model': args.flow_model,
                               'n_blocks': args.n_blocks,
                               'z_dim': args.z_dim,
                               'hidden_dim': args.hidden_dim,
                               'flow_hidden_size': args.flow_hidden_size,
                               'fixed_curvature': args.fixed_curvature,
                               'decoder': args.decoder,
                               'temperature': args.temperature,
                               'decoder': args.decoder
                               })
            opt.step()

        train_loss_avg[-1] /= len(train_loader.dataset)
        recon_loss_avg[-1] /= len(train_loader.dataset)
        kl_avg[-1] /= len(train_loader.dataset)
        print('Epoch: {:03d}, Pseudo KL:{:.4f}, Elbo: {:.4f}'.format(epoch,
                                                                     kl_avg[-1],
                                                                     train_loss_avg[-1]))
    return -1*train_loss.item(), model

def test_graph(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z, z_k = model.encode(x, train_pos_edge_index)
    return model.test(z_k, pos_edge_index, neg_edge_index)

def train_gae(args, opt, train_loader, test_loader, model):
    saved_path = './saved_models/' + args.dataset + '_' + args.model + '_' + str(args.z_dim) + '_gae.pt'
    cond = True if (os.path.exists(saved_path) and args.load_gae) else False
    if cond:
        model.load_no_flow(saved_path)
        return model

    train_loss_avg, recon_loss_avg, kl_avg = [], [], []
    num_batches, test_ll = 0, 0
    for epoch in range(0, args.epochs):
        train_loss_avg.append(0)
        recon_loss_avg.append(0)
        kl_avg.append(0)
        for batch_idx, data_batch in enumerate(train_loader):
            opt.zero_grad()
            if model.type != 'euclidean' and epoch < 10:
                model.encoder.radius.data = torch.ones_like(model.encoder.radius.data).to(torch.float64) * (11 - epoch)
            for idx, data_graph in enumerate(data_batch):
                data_graph.train_mask = data_graph.val_mask = data_graph.test_mask = data_graph.y = None
                data_graph.batch = None

                ''' Check if Data is split'''
                try:
                    x, edge_index = data_graph.x.to(args.dev), data_graph.edge_index.to(args.dev)
                    data = data_graph
                except:
                    data_graph.x.cuda()
                    x, edge_index = data.x.to(args.dev), data.edge_index.to(args.dev)

                if args.use_rand_feats:
                    x = perm_node_feats(x)

                z, z_k = model.encode(x, edge_index)
                recon_loss = model.recon_loss(z_k, edge_index)
                train_loss = recon_loss
                train_loss.backward()
                train_loss_avg[-1] += -1 * train_loss.item()
                recon_loss_avg[-1] += recon_loss.sum(dim=-1).item()

            # Do Step after each batch
            opt.step()

        ### Compute Epoch Stats
        train_loss_avg[-1] /= len(train_loader.dataset)
        recon_loss_avg[-1] /= len(train_loader.dataset)
        print('Epoch: {:03d}, Avg. Elbo:{:.4f}, Avg. Recon Loss:{:.4f}'.format(epoch, train_loss_avg[-1], recon_loss_avg[ -1]))
        ### Online Logging
        train_loss_metric = 'Train Loss'
        elbo_metric = 'Elbo'
        recon_loss_metric = 'Reconstruction'
        if args.wandb:
            wandb.log({train_loss_metric: -1 * train_loss_avg[-1], elbo_metric:
                       train_loss_avg[-1],recon_loss_metric:
                       recon_loss_avg[-1]})

    if not os.path.exists('./saved_models/'):
        os.makedirs('./saved_models/')

    model.save(saved_path)
    return model


def train_generation_wrapper(args, opt, train_loader, test_loader, model):
    if args.deterministic:
        gae_model = train_gae(args, opt, train_loader, test_loader, model)
        # No longer do we need to be deterministic
        gae_model.deterministic = False
    else:
        auc, trained_model = train_graph_generation(args, opt, train_loader,
                                         test_loader.dataset, model)
        flow_model = trained_model.encoder.flow_model
        test_metric = test_generation(args, test_loader, model.decoder,
                                      model.decoder_name, model.encoder.radius, flow_model)
        return test_metric, flow_model

    radius = torch.Tensor([args.radius]).to(args.dev)
    if args.flow_model:
        flow_model = kwargs_flows[args.flow_model](args.n_blocks, args.z_dim,
                                                   args.flow_hidden_size,
                                                   args.n_hidden,
                                                   layer_type=args.flow_layer_type,
                                                   radius=gae_model.encoder.radius).to(args.dev)
    else:
        test_metric = test_generation(args, test_loader, model.decoder,
                                      model.decoder_name, model.encoder.radius,
                                      None)
        return test_metric, model

    train_loss_avg = []
    preclamp_cond = (args.flow_model and args.model.lower() != 'euclidean')
    flow_opt = optim.Adam(flow_model.parameters(), lr=args.flow_lr)
    for epoch in range(0, args.flow_epochs):
        train_loss_avg.append(0)
        for batch_idx, data_batch in enumerate(train_loader):
            flow_opt.zero_grad()
            for idx, data_graph in enumerate(data_batch):

                ''' Check if Data is split'''
                try:
                    x, edge_index = data_graph.x.to(args.dev), data_graph.edge_index.to(args.dev)
                    data = data_graph
                except:
                    data_graph.x.cuda()

                # This is our new input
                with torch.no_grad():
                    z, _ = gae_model.encode(x, edge_index)
                    if args.model != 'euclidean':
                        flow_model.base_dist_mean = gae_model.__mu_h__
                        flow_model.base_dist_var = gae_model.__std__
                    else:
                        flow_model.base_dist_mean = gae_model.__mu__
                        flow_model.base_dist_var = F.softplus(gae_model.__logvar__) + 1e-5

                loss = -1 * flow_model.log_prob(z, edge_index).mean()
                loss.backward()
                train_loss_avg[-1] += loss.item()

            # Do Step after each batch
            flow_opt.step()

        train_loss_avg[-1] /= len(train_loader.dataset)
        preclamp_norm = flow_model.preclamp_norm.item() if preclamp_cond else 0
        print('Flow Epoch: {:03d}, Avg. Loss:{:.4f}, Pre-Clamp Norm: {:.4f}'.format(epoch, train_loss_avg[-1], preclamp_norm))

        ### Online Logging
        train_loss_metric = 'Flow Train Loss'
        if args.wandb:
            wandb.log({train_loss_metric: train_loss_avg[-1], "Pre-Clamp Norm": preclamp_norm})

    test_metric = test_generation(args, test_loader, gae_model.decoder,
                                  gae_model.decoder_name,
                                  gae_model.encoder.radius, flow_model)
    return test_metric, flow_model


def test_generation(args, test_loader, decoder, decoder_name, radius, flow_model=None):
    node_dist = args.node_dist
    det_name = ''
    edge_index = None
    flow_name = ''

    if args.deterministic:
        det_name = 'deterministic'

    if args.flow_model:
        flow_name = args.flow_model

    if not decoder_name:
        decoder_name = ''

    save_gen_base = plots = './visualization/gen_plots/' + args.dataset + '/'
    save_gen_plots = save_gen_base + args.model + str(args.z_dim) + '_' \
        + flow_name + '_' + decoder_name + '_' + det_name + '/'
    gen_graph_list, gen_graph_copy_list = [], []
    avg_connected_components, avg_triangles, avg_transitivity = [], [], []
    raw_triangles = []
    for i in range(0, args.num_gen_samples):
        nodes2gen = np.random.choice(node_dist, 1)[0]
        node_embed_shape = torch.Size([nodes2gen, args.z_dim])
        num_samples = torch.Size([1])
        fully_connected = nx.complete_graph(nodes2gen)
        edge_index = torch.tensor(list(fully_connected.edges)).to(args.dev).t().contiguous()
        if flow_model is None:
            if args.model == 'euclidean':
                prior = EuclideanNormal(torch.zeros(node_embed_shape, device=args.dev),
                                        torch.ones(node_embed_shape, device=args.dev))
            else:
                mu_0_shape = torch.Size([nodes2gen, args.z_dim + 1])
                prior = HyperboloidWrappedNormal(radius, torch.zeros(mu_0_shape, device=args.dev),
                                                 torch.ones(node_embed_shape, device=args.dev))
            z_k = prior.rsample(num_samples).squeeze()
        else:
            # if flow_model.layer_type != 'Linear':
            if args.model == 'euclidean':
                prior = EuclideanNormal(torch.zeros(node_embed_shape, device=args.dev),
                                        torch.ones(node_embed_shape, device=args.dev))
            else:
                mu_0_shape = torch.Size([nodes2gen, args.z_dim + 1])
                prior = HyperboloidWrappedNormal(flow_model.radius, torch.zeros(mu_0_shape, device=args.dev),
                                                 torch.ones(node_embed_shape, device=args.dev))

            z_0 = prior.rsample(num_samples).squeeze()
            z_k, _ = flow_model.inverse(z_0, edge_index)
        if args.model != 'euclidean' and decoder_name not in ['fermi', 'tanh',
                                                              'distance',
                                                              'softmax']:
            # Log-map z back to \mathcal{T}_{\textbf{o}}\mathbb{H}
            z_k = inverse_exp_map_mu0(z_k, radius)

        adj = decoder.forward_all(z_k, edge_index)
        adj_mat = torch.bernoulli(adj)
        G = nx.from_numpy_matrix(adj_mat.detach().cpu().numpy())
        G.remove_edges_from(nx.selfloop_edges(G))
        # num_connected_components = nx.number_connected_components(G)
        # avg_connected_components.append(num_connected_components)
        # num_triangles = list(nx.triangles(G).values())
        # avg_triangles.append(sum(num_triangles) / float(len(num_triangles)))
        # avg_transitivity.append(nx.transitivity(G))
        # raw_triangles.append([num_triangles, len(G.nodes)])
        G_copy = copy.deepcopy(G)
        gen_graph_copy_list.append(G_copy)
        G.remove_nodes_from(list(nx.isolates(G)))
        if len(G) > 0:
            G = max(nx.connected_component_subgraphs(G), key=len)
            num_connected_components = nx.number_connected_components(G)
            avg_connected_components.append(num_connected_components)
            num_triangles = list(nx.triangles(G).values())
            avg_triangles.append(sum(num_triangles) / float(len(num_triangles)))
            avg_transitivity.append(nx.transitivity(G))
            raw_triangles.append([num_triangles, len(G.nodes)])
            draw_nx_graph(G, name=args.dataset + '_' + str(i), path=save_gen_plots)
            gen_graph_list.append(G)

    # Evaluate Generated Graphs using GraphRNN metrics
    test_dataset = [to_networkx(test_G).to_undirected() for test_G in test_loader.dataset]
    metrics = evaluate_generated(test_dataset, gen_graph_list, args.dataset)
    metrics_copy = evaluate_generated(test_dataset, gen_graph_copy_list, args.dataset)
    # Orginal Graphs with nodes remoed
    mmd_degree, mmd_clustering, mmd_4orbits = metrics[0], metrics[1], metrics[2]
    mmd_spectral, accuracy = metrics[3], metrics[4]
    mean_connected_comps = sum(avg_connected_components) / float(len(avg_connected_components))
    mean_triangles = sum(avg_triangles) / float(len(avg_triangles))
    mean_transitivity = sum(avg_transitivity) / float(len(avg_transitivity))

    # Copied Graphs with nodes not removed
    mmd_degree_copy, mmd_clustering_copy, mmd_4orbits_copy = metrics_copy[0], metrics_copy[1], metrics_copy[2]
    mmd_spectral_copy, accuracy_copy = metrics_copy[3], metrics_copy[4]
    if args.wandb:
        wandb.log({"Deg": mmd_degree, "Clus": mmd_clustering, "Orb":
                   mmd_4orbits, "Acc": accuracy, "Spec.": mmd_spectral,
                   "Avg_CC": mean_connected_comps, "Avg_Tri": mean_triangles,
                   "Avg_transitivity": mean_transitivity, "Raw_triangles":
                   raw_triangles})
        wandb.log({"Deg_copy": mmd_degree_copy, "Clus_copy":
                   mmd_clustering_copy,  "Orb_copy": mmd_4orbits_copy,
                   "Acc_copy": accuracy_copy, "Spec_copy": mmd_spectral_copy})

    print('Deg: {:.4f}, Clus.: {:.4f}, Orbit: {:.4f}, Spec.:{:.4f}, Acc: {:.4f}'.format(mmd_degree, \
                                                                           mmd_clustering,
                                                                           mmd_4orbits,
                                                                           mmd_spectral,
                                                                           accuracy))
    print('Avg CC: {:.4f}, Avg. Tri: {:.4f}, Avg. Trans: {:.4f}'.format(mean_connected_comps, mean_triangles,
          mean_transitivity))
    return [mmd_degree, mmd_clustering, mmd_4orbits, mmd_spectral, accuracy]
