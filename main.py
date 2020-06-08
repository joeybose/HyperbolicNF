import argparse
import json
import os
import wandb
import torch
from torch import optim

from data import create_dataset
from models import create_model
from train_helper import train_wrapper
from utils import utils
from utils.utils import seed_everything, str2bool

def main(args):
    dataset = create_dataset(args, args.dataset, args.batch_size, args.data)
    train_loader, test_loader = dataset.create_loaders()
    flow_args = [args.n_blocks, args.flow_hidden_size, args.n_hidden,
                 args.flow_model, args.flow_layer_type]
    model = create_model(args.model.lower(), args.deterministic,
                         args.enc_blocks, args.conv_type, args.dataset.lower(),
                         args.decoder, args.r, args.temperature, args.beta,
                         args.input_dim, args.hidden_dim, args.z_dim,
                         dataset.reconstruction_loss, args.ll_estimate, args.K,
                         flow_args, args.dev, args.fixed_curvature,
                         args.radius).to(args.dev)
    print(vars(args))
    print(f"VAE Model: {model}; Epochs: {args.epochs}; Dataset: {dataset}; ")
    ### Start Training ###
    opt = optim.Adam(model.parameters(), lr=args.lr)
    train_metric, model = train_wrapper(args, opt, train_loader, test_loader, model)


if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Euclidean')
    parser.add_argument('--conv_type', type=str, default='GCN', help='VGAE Conv Layer')
    parser.add_argument('--node_order', type=str, default='DFS', help='Node Order to use for VGAE during Generation')
    parser.add_argument('--input_dim', type=int, default='784',
                        help='This will be set in create_datasets automatically')
    parser.add_argument('--hidden_dim', type=int, default='200')
    parser.add_argument('--z_dim', type=int, default='2')
    parser.add_argument('--radius', type=float, default='1')
    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--flow_lr', type=float, default='1e-3')
    parser.add_argument("--fixed_curvature",
                        type=str2bool,
                        default=True,
                        help="Whether to fix curvatures to radius.")
    parser.add_argument("--deterministic",
                        type=str2bool,
                        default=False,
                        help="Whether to use GAE or VGAE.")
    parser.add_argument("--load_gae",
                        type=str2bool,
                        default=True,
                        help="Whether to load pretrained GAE model.")
    parser.add_argument("--data", type=str, default="./data", help="Data directory.")
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--flow_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--eval_set', default="test",
                        help="Whether to evaluate model on test set (default) or validation set.")
    parser.add_argument('--val_edge_ratio', type=float, default='0.05')
    parser.add_argument('--test_edge_ratio', type=float, default='0.1')
    parser.add_argument('--decoder', type=str, default=None)
    parser.add_argument('--r', type=float, default='2', help='Hyperparameter for Fermi-Dirac Decoder')
    parser.add_argument('--num_fixed_features', default=10, type=int)
    parser.add_argument('--edge_threshold', default=0.75, type=float)
    parser.add_argument('--temperature', type=float, default='1',
                        help='Hyperparameter for Fermi-Dirac Decoder')
    parser.add_argument("--normalize_adj",
                        type=str2bool,
                        default=False,
                        help="whether to row-normalize the adjacency matrix")
    parser.add_argument("--normalize_feats",
                        type=str2bool,
                        default=False,
                        help="whether to normalize input node features")
    parser.add_argument("--use_rand_feats",
                        type=str2bool,
                        default=False,
                        help="whether to use Random Node features for graphs")
    parser.add_argument('--kl_anneal', type=int, default=0, help='KL Anneal Coefficient')
    parser.add_argument('--ll_estimate', type=str, default='mc',
                        help='Type of log likelihood estimate: mc (Monte Carlo)'
                        'or iwae (Importance Weighted Autoencoders')
    parser.add_argument('--K', type=int, default=500, help='Number of samples for log likelihood estimate.')
    parser.add_argument('--num_gen_samples', type=int, default=100,
                        help='Number of Graphs to Generate')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta-VAE param')
    parser.add_argument('--flow_hidden_size', type=int, default=128, \
                        help='Hidden layer size for Flows.')
    parser.add_argument('--n_blocks', type=int, default=2, \
                        help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
    parser.add_argument('--enc_blocks', type=int, default=0, \
                        help='Number of Additional blocks in VGAE Encoder.')
    parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each Flow.')
    parser.add_argument('--num_channels', type=int, default=16, help='Number of channels in VGAE.')
    parser.add_argument('--flow_model', default=None, help='Which model to use')
    parser.add_argument('--flow_layer_type', type=str, default='Linear',
                        help='Which type of layer to use ---i.e. GRevNet or Linear')
    parser.add_argument('--do_kl_anneal', action="store_true", default=False, help='Do KL Annealing')
    parser.add_argument("--clip_grad", action="store_true", default=False, help='Gradient Clipping')
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--seed', type=int, metavar='S', help='random seed (default: None)')
    parser.add_argument('--debug', default=False, action='store_true', help='Debug')
    parser.add_argument('--namestr', type=str, default='Floss',
                        help='additional info in output filename to describe experiments')
    args = parser.parse_args()

    ''' Fix Random Seed '''
    seed_everything(args.seed)
    # Check if settings file
    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_name = utils.project_name(args.dataset)

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='metrics-{}-{}'.format(project_name, args.eval_set), name=args.namestr)

    main(args)
