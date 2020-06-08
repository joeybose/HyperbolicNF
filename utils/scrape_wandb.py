import argparse
import csv
import json
import os
from statistics import mean
import wandb
import matplotlib
import numpy as np
from utils import project_name
import ipdb


def get_data(args, wandb_username, wandb_project):
    # Accumulate data for all experiments.
    data_experiments_auc = []
    data_experiments_ap = []
    data_experiments_metric = []
    data_runs_auc = []
    data_runs_ap = []
    data_runs_metric = []
    runs = list(args.api.runs(wandb_username+ "/"+ wandb_project))
    if args.config_key:

        filters = list(zip(args.config_key, args.config_val))

        def keep_run(run):
            for (k, v) in filters:
                typed_val = type(run.config.get(k))
                if typed_val != v:
                    return False
            else:
                return True

        filtered_runs = filter(keep_run, runs)

    else:
        filtered_runs = [single_run for single_run in runs if args.name_str ==
                         single_run.name]
    # Filter out crashed runs
    filtered_runs = [single_run for single_run in filtered_runs if 'finished' in
                     single_run.state]

    # Accumulate data for all runs of a given project
    for my_run in filtered_runs:
        raw_data = my_run.history(samples=100000)
        keys = raw_data.keys().values
        if not args.custom_metric:
            test_keys = [key for key in keys if 'Test' in key]
        else:
            test_keys = [key for key in keys if args.metric in key]
        for key in test_keys:
            if 'AUC' in key:
                data_points_auc = raw_data[key].dropna().values
                data_experiments_auc.append(data_points_auc)
            elif 'AP' in key:
                data_points_ap = raw_data[key].dropna().values
                data_experiments_ap.append(data_points_ap)
            elif args.metric in key:
                # data_points_metric = raw_data[key].dropna().values
                data_points_metric = raw_data[args.metric].dropna().values
                data_experiments_metric.append(data_points_metric)

    last_data_points_auc = [data_run[-1] for data_run in data_experiments_auc]
    last_data_points_ap = [data_run[-1] for data_run in data_experiments_ap]
    last_data_points_metric = [data_run[-1] for data_run in data_experiments_metric]
    return last_data_points_auc, last_data_points_ap, last_data_points_metric

def main(args):
    # wandb_project = '{}-{}'.format(project_name(args.dataset),args.eval_set)
    wandb_project = 'metrics-{}-{}'.format(project_name(args.dataset),args.eval_set)
    wandb_username = args.wandb_uname
    if args.graph_gen:
        args.custom_metric = True
        args.metric = 'Deg'
        # args.metric = 'Deg_copy'
        auc, ap, deg_metric = get_data(args, wandb_project=wandb_project,
                                   wandb_username=wandb_username)
        deg_metric = np.sort(deg_metric)[::-1]
        mean_deg_metric, std_deg_metric = np.mean(deg_metric[0:args.top_k]), np.std(deg_metric[0:args.top_k])
        args.metric = 'Clus'
        # args.metric = 'Clus_copy'
        auc, ap, clus_metric = get_data(args, wandb_project=wandb_project,
                                   wandb_username=wandb_username)
        clus_metric = np.sort(clus_metric)[::-1]
        mean_clus_metric, std_clus_metric = np.mean(clus_metric[0:args.top_k]), np.std(clus_metric[0:args.top_k])
        args.metric = 'Orb'
        # args.metric = 'Orb_copy'
        auc, ap, orb_metric = get_data(args, wandb_project=wandb_project,
                                   wandb_username=wandb_username)
        orb_metric = np.sort(orb_metric)[::-1]
        mean_orb_metric, std_orb_metric = np.mean(orb_metric[0:args.top_k]), np.std(orb_metric[0:args.top_k])
        args.metric = 'Spec.'
        # args.metric = 'Spec_copy'
        auc, ap, spec_metric = get_data(args, wandb_project=wandb_project,
                                   wandb_username=wandb_username)
        spec_metric = np.sort(spec_metric)[::-1]
        mean_spec_metric, std_spec_metric = np.mean(spec_metric[0:args.top_k]), np.std(spec_metric[0:args.top_k])
        args.metric = 'Acc'
        # args.metric = 'Acc_copy'
        auc, ap, acc_metric = get_data(args, wandb_project=wandb_project,
                                   wandb_username=wandb_username)
        acc_metric = np.sort(acc_metric)[::-1]
        mean_acc_metric, std_acc_metric = np.mean(acc_metric[0:args.top_k]), np.std(acc_metric[0:args.top_k])
        args.metric = 'Avg_CC'
        auc, ap, cc_metric = get_data(args, wandb_project=wandb_project,
                                   wandb_username=wandb_username)
        cc_metric = np.sort(cc_metric)
        mean_cc_metric, std_cc_metric = np.mean(cc_metric[0:args.top_k]), np.std(cc_metric[0:args.top_k])

        args.metric = 'Avg_Tri'
        auc, ap, tri_metric = get_data(args, wandb_project=wandb_project,
                                   wandb_username=wandb_username)
        tri_metric = np.sort(tri_metric)
        mean_tri_metric, std_tri_metric = np.mean(tri_metric[0:args.top_k]), np.std(tri_metric[0:args.top_k])

        args.metric = 'Avg_transitivity'
        auc, ap, trans_metric = get_data(args, wandb_project=wandb_project,
                                   wandb_username=wandb_username)
        trans_metric = np.sort(trans_metric)
        mean_trans_metric, std_trans_metric = np.mean(trans_metric[0:args.top_k]), np.std(trans_metric[0:args.top_k])

        print("Test Mean: %s %f, Std: %f" % ("Deg", mean_deg_metric, std_deg_metric))
        print("Test Mean: %s %f, Std: %f" % ("Clus", mean_clus_metric, std_clus_metric))
        print("Test Mean: %s %f, Std: %f" % ("Orb", mean_orb_metric, std_orb_metric))
        print("Test Mean: %s %f, Std: %f" % ("Spec.", mean_spec_metric, std_spec_metric))
        print("Test Mean: %s %f, Std: %f" % ("Acc", mean_acc_metric, std_acc_metric))
        print("Test Mean: %s %f, Std: %f" % ("Avg CC", mean_cc_metric, std_cc_metric))
        print("Test Mean: %s %f, Std: %f" % ("Avg Tri", mean_tri_metric, std_tri_metric))
        print("Test Mean: %s %f, Std: %f" % ("Avg Trans", mean_trans_metric, std_trans_metric))
        return None
    else:
        auc, ap, metric = get_data(args, wandb_project=wandb_project,
                                   wandb_username=wandb_username)

    if len(auc) > 0:
        auc = np.sort(auc)[::-1]
        if len(auc) > args.top_k:
            mean_auc, std_auc = np.mean(auc[0:args.top_k]), np.std(auc[0:args.top_k])
        else:
            mean_auc, std_auc = np.mean(auc), np.std(auc)
        print("Test Mean: AUC %f, Std: %f" % (mean_auc, std_auc))

    if len(ap) > 0:
        ap = np.sort(ap)[::-1]
        if len(ap) > args.top_k:
            mean_ap, std_ap = np.mean(ap[0:args.top_k]), np.std(ap[0:args.top_k])
        else:
            mean_ap, std_ap = np.mean(ap), np.std(ap)
        print("Test Mean: AP %f, Std: %f" % (mean_ap, std_ap))

    if len(metric) > 0:
        metric = np.sort(metric)[::-1]
        if len(metric) > args.top_k:
            mean_metric, std_metric = np.mean(metric[0:args.top_k]), np.std(metric[0:args.top_k])
        else:
            mean_metric, std_metric = np.mean(metric), np.std(metric)
        print("Test Mean: %s %f, Std: %f" % (args.metric, mean_metric, std_metric))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='Train')
    parser.add_argument('--name_str', type=str, default='')
    parser.add_argument('--wandb_uname', type=str, default='')
    parser.add_argument('--metric', type=str, default='Test MC Log Likelihood')
    parser.add_argument('--custom_metric', action="store_true", default=False,
                        help='Custom non Test metric')
    parser.add_argument('--graph_gen', action="store_true", default=False,
                        help='Report all graph gen Test metric')
    parser.add_argument('--top_k', type=int, default=5, help='Return only top K runs')
    parser.add_argument('--config_key', nargs='*', type=str, default=[])
    parser.add_argument('--config_val', nargs='*', default=[])
    parser.add_argument('--dataset', type=str, default='bdp')
    parser.add_argument('--eval_set', default="test",
                        help="Whether to evaluate model on test set (default) or validation set.")
    parser.add_argument('--get_step', nargs='+', default=5, type=int)
    args = parser.parse_args()

    with open('../settings.json') as f:
        data = json.load(f)
    args.wandb_apikey = data.get("wandbapikey")
    os.environ['WANDB_API_KEY'] = args.wandb_apikey
    args.api = wandb.Api()

    main(args)

