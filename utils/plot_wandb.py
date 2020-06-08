import argparse
import csv
import json
import os
from statistics import mean
import wandb
import matplotlib
import numpy as np
import ipdb
os.environ['WANDB_API_KEY'] = "7110d81f721ee9a7da84c67bcb319fc902f7a180"
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
api = wandb.Api()

# Set plotting style
sns.set_context('paper', font_scale=1.3)
sns.set_style('whitegrid')
sns.set_palette('colorblind')
plt.rcParams['text.usetex'] = False

def SetPlotRC():
    #If fonttype = 1 doesn't work with LaTeX, try fonttype 42.
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)

def ApplyFont(ax):

    ticks = ax.get_xticklabels() + ax.get_yticklabels()

    text_size = 14.0

    for t in ticks:
        t.set_fontname('Times New Roman')
        t.set_fontsize(text_size)

    txt = ax.get_xlabel()
    txt_obj = ax.set_xlabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_ylabel()
    txt_obj = ax.set_ylabe(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_title()
    txt_obj = ax.set_title(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

SetPlotRC()

def data_to_extract(username,args):
    labels = {}
    labels['title'] = "PPI Link Prediction"
    labels['x_label'] = "Gradient Updates"
    labels['y_label'] = "Percent"
    if args.local:
        param_str = 'Local'
    else:
        param_str = 'Global'

    labels['train_metric_auc'] = "Train_" + param_str + "_Graph_"
    labels['train_metric_ap'] = "Train_" + param_str + "_Graph_"
    labels['test_metric_auc'] = "Test_" + param_str + "_Graph_"
    labels['test_metric_ap'] = "Test_" + param_str + "_Graph_"
    if username == "joeybose":
      labels['experiments_key'] = [[args.one_maml],\
                                   [args.two_maml],\
                                   [args.random_exp],\
                                   [args.no_finetune],\
                                   [args.finetune],\
                                   [args.adamic_adar],\
                                   [args.graph_sig],\
                                   ]
    if args.local:
        labels['experiments_name'] = ['1-MAML','2-MAML', 'NoFinetune',\
                  'Finetune','Adamic-Adar']
    else:
        labels['experiments_name'] = ['1-MAML','2-MAML', 'Random', 'NoFinetune',\
              'Finetune']

    return labels

def data_to_extract_ppi(username,args):
    labels = {}
    labels['title'] = "PPI Link Prediction"
    labels['x_label'] = "Gradient Updates"
    labels['y_label'] = "Percent"
    if args.local:
        param_str = 'Local'
    else:
        param_str = 'Global'

    labels['train_metric_auc'] = "Train_" + param_str + "_Graph_"
    labels['train_metric_ap'] = "Train_" + param_str + "_Graph_"
    labels['test_metric_auc'] = "Test_" + param_str + "_Graph_"
    labels['test_metric_ap'] = "Test_" + param_str + "_Graph_"
    if username == "joeybose":
      labels['experiments_key'] = [[args.two_maml],\
                                   [args.concat],\
                                   [args.random_exp],\
                                   [args.no_finetune],\
                                   [args.finetune],\
                                   [args.adamic_adar],\
                                   [args.mlp],\
                                   [args.graph_sig],\
                                   [args.graph_sig_concat],\
                                   [args.graph_sig_sig],\
                                   [args.graph_sig_weights],\
                                   [args.graph_sig_random]\
                                   ]

    labels['experiments_name'] = ['2-MAML', 'MAML','Random', 'NoFinetune',\
          'Finetune', 'Adamic-Adar', 'MLP', 'Graph-Sig', 'Graph-Sig-Concat',\
                                  'GS-Gating','GS-Weights', 'Graph-Sig-Random']

    if args.local:
        if args.local_block is not None:
            for block in sorted(args.local_block, reverse=True):
                del labels['experiments_name'][block]
                del labels['experiments_key'][block]
    else:
        if args.global_block is not None:
            for block in sorted(args.global_block, reverse=True):
                del labels['experiments_name'][block]
                del labels['experiments_key'][block]
    return labels

def data_to_extract_firstmmdb(username,args):
    labels = {}
    labels['title'] = "FIRSTMMDB Link Prediction"
    labels['x_label'] = "Gradient Updates"
    labels['y_label'] = "AUC/AP"
    if args.local:
        param_str = 'Local'
    else:
        param_str = 'Global'

    labels['train_metric_auc'] = "Train_" + param_str + "_Graph_"
    labels['train_metric_ap'] = "Train_" + param_str + "_Graph_"
    labels['test_metric_auc'] = "Test_" + param_str + "_Graph_"
    labels['test_metric_ap'] = "Test_" + param_str + "_Graph_"
    if username == "joeybose":
      labels['experiments_key'] = [[args.two_maml],\
                                   [args.concat],\
                                   [args.random_exp],\
                                   [args.no_finetune],\
                                   [args.finetune],\
                                   [args.adamic_adar],\
                                   [args.mlp],\
                                   [args.graph_sig],\
                                   [args.graph_sig_sig],\
                                   [args.graph_sig_weights],\
                                   [args.graph_sig_concat],\
                                   [args.graph_sig_random]\
                                   ]

    labels['experiments_name'] = ['MAML', 'MAML-Concat','Random', 'NoFinetune',\
          'Finetune', 'Adamic-Adar', 'MLP', 'Graph-Sig', 'GS-Gating',\
                                  'GS-Weights','Graph-Sig Concat','Graph-Sig-Random']

    if args.local:
        if args.local_block is not None:
            for block in sorted(args.local_block, reverse=True):
                del labels['experiments_name'][block]
                del labels['experiments_key'][block]
    else:
        if args.global_block is not None:
            for block in sorted(args.global_block, reverse=True):
                del labels['experiments_name'][block]
                del labels['experiments_key'][block]
    return labels

def data_to_extract_reddit(username,args):
    labels = {}
    labels['title'] = "Reddit Link Prediction"
    labels['x_label'] = "Gradient Updates"
    labels['y_label'] = "Percent"
    if args.local:
        param_str = 'Local_Batch'
    else:
        param_str = 'Global_Batch'

    labels['train_metric_auc'] = "Train_" + param_str + "_Graph_"
    labels['train_metric_ap'] = "Train_" + param_str + "_Graph_"
    labels['test_metric_auc'] = "Test_" + param_str + "_Graph_"
    labels['test_metric_ap'] = "Test_" + param_str + "_Graph_"
    if username == "joeybose":
      labels['experiments_key'] = [[args.two_maml],\
                                   [args.random_exp],\
                                   [args.no_finetune],\
                                   [args.finetune],\
                                   [args.adamic_adar],\
                                   [args.mlp],\
                                   [args.graph_sig]\
                                   ]
    labels['experiments_name'] = ['2-MAML', 'Random', 'NoFinetune',\
          'Finetune', 'Adamic-Adar', 'MLP', 'Graph-Sig']

    if args.local:
        for block in sorted(args.local_block, reverse=True):
            del labels['experiments_name'][block]
            del labels['experiments_key'][block]
    else:
        for block in sorted(args.global_block, reverse=True):
            del labels['experiments_name'][block]
            del labels['experiments_key'][block]

    return labels

def truncate_exp(data_experiments):
    last_data_points = [run[-1] for data_run in data_experiments for run in data_run]
    run_end_times = [timestep for timestep, value in last_data_points]
    earliest_end_time = min(run_end_times)

    clean_data_experiments = []
    for exp in data_experiments:
        clean_data_runs = []
        for run in exp:
            clean_data_runs.append({x: y for x, y in run if x <= earliest_end_time})
        clean_data_experiments.append(clean_data_runs)

    return clean_data_experiments

def truncate_array(experiment_list):
    min_len = 10000000
    for exp in experiment_list:
        length = exp.shape[1]
        if length < min_len:
            min_len = length
    trunc_exp_list = []
    for exp in experiment_list:
        trunc_exp = np.vstack((exp[0][:min_len],exp[1][:min_len])).T
        trunc_exp_list.append(trunc_exp)
    return trunc_exp_list

def get_grad_step_data(args, labels, wandb_username, wandb_project):
    # Accumulate data for all experiments.
    data_experiments_auc = []
    data_experiments_ap = []
    for runs, label in zip(labels.get('experiments_key'), labels.get('experiments_name')):
        if len(runs[0]) > 0:
            for exp_key in runs:
                try:
                    my_run = api.run("%s/%s/%s" %(wandb_username,\
                            wandb_project,exp_key))
                except:
                    temp_wandb_project='meta-graph'
                    my_run = api.run("%s/%s/%s" %(wandb_username,\
                            temp_wandb_project,exp_key))
                raw_data = my_run.history(samples=1000000)
                keys = raw_data.keys().values
                test_inner_avg_auc = 'Test_Complete_AUC'
                test_inner_avg_ap = 'Test_Complete_AP'
                test_avg_auc = 'Test_Avg__AUC'
                test_avg_ap = 'Test_Avg__AP'
                try:
                    data_points_auc = raw_data[test_inner_avg_auc].dropna().values
                    data_points_ap = raw_data[test_inner_avg_ap].dropna().values
                    data_points_auc_array = [[i,point] for i, point in enumerate(data_points_auc)]
                    data_points_ap_array = [[i,point] for i, point in enumerate(data_points_ap)]
                    data_experiments_auc.append(np.asarray(data_points_auc_array).T)
                    data_experiments_ap.append(np.asarray(data_points_ap_array).T)
                except:
                    data_points_auc = raw_data[test_avg_auc].dropna().values
                    data_points_ap = raw_data[test_avg_ap].dropna().values
                    data_points_auc_array = [[i,point] for i, point in enumerate(data_points_auc)]
                    data_points_ap_array = [[i,point] for i, point in enumerate(data_points_ap)]
                    data_experiments_auc.append(np.asarray(data_points_auc_array).T)
                    data_experiments_ap.append(np.asarray(data_points_ap_array).T)

            print("%s AUC %f After %d Grad Steps"%(my_run.name,data_points_auc[int(args.get_step[0]/5)], args.get_step[0]))
            print("%s AP %f After %d Grad Steps"%(my_run.name,data_points_ap[int(args.get_step[0]/5)], args.get_step[0]))
            print("Max AUC %f"%(data_points_auc.max()))
            print("Max AP %f"%(data_points_ap.max()))
    clean_data_experiments_auc = truncate_array(data_experiments_auc)
    clean_data_experiments_ap = truncate_array(data_experiments_ap)
    return clean_data_experiments_auc, clean_data_experiments_ap

def get_data(args, title, x_label, y_label, labels_list, data, wandb_username, wandb_project):
    if not title or not x_label or not y_label or not labels_list:
        print("Error!!! Ensure filename, x and y labels,\
        and metric are present.")
        exit(1)

    train_auc = labels_list['train_metric_auc']
    train_ap = labels_list['train_metric_ap']
    test_auc = labels_list['train_metric_auc']
    test_ap = labels_list['train_metric_ap']

    # Accumulate data for all experiments.
    data_experiments_auc = []
    data_experiments_ap = []
    for i, runs in enumerate(data):
        # Accumulate data for all runs of a given experiment.
        data_runs_auc = []
        data_runs_ap = []
        if len(runs) > 0:
            for exp_key in runs:
                try:
                    my_run = api.run("%s/%s/%s" %(wandb_username,\
                            wandb_project,exp_key))
                except:
                    wandb_project='meta-graph-ppi'
                    my_run = api.run("%s/%s/%s" %(wandb_username,\
                            wandb_project,exp_key))
                raw_data = my_run.history(samples=100000)
                keys = raw_data.keys().values
                train_keys = [key for key in keys if 'Train' in key]
                test_keys = [key for key in keys if 'Test' in key]
                train_local_keys = [key for key in train_keys if 'Local' in key]
                train_global_keys = [key for key in train_keys if 'Local' not in key]
                train_global_keys = [key for key in train_global_keys if 'Complete' not in key]
                train_global_keys = [key for key in train_global_keys if 'Avg' not in key]
                test_local_keys = [key for key in test_keys if 'Local' in key]
                test_global_keys = [key for key in test_keys if 'Local' not in key]
                test_global_keys = [key for key in test_global_keys if 'Complete' not in key]
                test_global_keys = [key for key in test_global_keys if 'Avg' not in key]

                ''' Add Random back in '''
                if (args.dataset == 'PPI' or args.dataset =='FIRSTMM_DB')  and i == 2 and args.local:
                    train_local_keys = train_global_keys
                    test_local_keys = test_global_keys
                elif (args.dataset == 'PPI' or args.dataset=='FIRSTMM_DB') and i == 9 and args.local:
                    train_local_keys = train_global_keys
                    test_local_keys = test_global_keys

                if args.mode == 'Train':
                    if args.local:
                        for key in train_local_keys:
                            if 'AUC' in key:
                                data_points_auc = raw_data[key].dropna().values
                                data_points_auc = [[i+1,point] for i, point in enumerate(data_points_auc)]
                                data_runs_auc.append(data_points_auc)
                            else:
                                data_points_ap = raw_data[key].dropna().values
                                data_points_ap = [[i+1,point] for i, point in enumerate(data_points_ap)]
                                data_runs_ap.append(data_points_ap)
                    else:
                        for key in train_global_keys:
                            if 'AUC' in key:
                                data_points_auc = raw_data[key].dropna().values
                                data_points_auc = [[i+1,point] for i, point in enumerate(data_points_auc)]
                                data_runs_auc.append(data_points_auc)
                            else:
                                data_points_ap = raw_data[key].dropna().values
                                data_points_ap = [[i+1,point] for i, point in enumerate(data_points_ap)]
                                data_runs_ap.append(data_points_ap)
                else:
                    if args.local:
                        for key in test_local_keys:
                            if 'AUC' in key:
                                data_points_auc = raw_data[key].dropna().values
                                data_points_auc = [[i,point] for i, point in enumerate(data_points_auc)]
                                data_runs_auc.append(data_points_auc)
                            else:
                                data_points_ap = raw_data[key].dropna().values
                                data_points_ap = [[i,point] for i, point in enumerate(data_points_ap)]
                                data_runs_ap.append(data_points_ap)
                    else:
                        for key in test_global_keys:
                            if 'AUC' in key:
                                data_points_auc = raw_data[key].dropna().values
                                data_points_auc = [[i,point] for i, point in enumerate(data_points_auc)]
                                data_runs_auc.append(data_points_auc)
                            else:
                                data_points_ap = raw_data[key].dropna().values
                                data_points_ap = [[i,point] for i, point in enumerate(data_points_ap)]
                                data_runs_ap.append(data_points_ap)
            data_experiments_auc.append(data_runs_auc)
            data_experiments_ap.append(data_runs_ap)
    clean_data_experiments_auc = truncate_exp(data_experiments_auc)
    clean_data_experiments_ap = truncate_exp(data_experiments_ap)
    return clean_data_experiments_auc, clean_data_experiments_ap

def my_plot(**kwargs):
    labels = kwargs.get('labels')
    data = kwargs.get('data')

    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()

    for label in (ax.get_xticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(20)
    for label in (ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.yticks(np.arange(0.5, 1, 0.05))
    ax.set_yticks(np.arange(0.5, 1, 0.05))
    # plt.xticks(np.arange(0, 200, 5))
    ax.xaxis.get_offset_text().set_fontsize(5)
    ax.yaxis.get_offset_text().set_fontsize(5)
    axis_font = {'fontname': 'Arial', 'size': '24'}
    colors = sns.color_palette('colorblind', n_colors=len(data))

    # Plot data
    if args.mode=='Test':
        position = 0

    my_labels = []
    for label,key in zip(labels.get('experiments_name'),labels.get('experiments_key')):
        if len(key[0]) > 0:
            my_labels.append(label)

    for runs, label, color in zip(data, my_labels, colors):
        unique_x_values = set()

        # Plot mean and standard deviation of all runs
        y_values_mean = []
        y_values_std = []
        runs = np.asarray(runs).T
        x_values = runs[0]
        y_values = runs[1]
        # Plot mean
        plt.plot(x_values, y_values, color=color, linewidth=2.5, label=label)

    # Label figure
    ax.legend(loc='lower right', prop={'size': 16})
    if args.mode == 'Train':
        ax.set_xlabel(labels.get('x_label'), **axis_font)
    ax.set_ylabel(labels.get('y_label'), **axis_font)
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.2)

    # remove grid lines
    ax.grid(False)
    plt.grid(b=False, color='w')
    return fig

def plot(**kwargs):
    labels = kwargs.get('labels')
    data = kwargs.get('data')

    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()

    for label in (ax.get_xticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(20)
    for label in (ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(20)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.yticks(np.arange(0, 1, 0.1))
    ax.xaxis.get_offset_text().set_fontsize(10)
    axis_font = {'fontname': 'Arial', 'size': '24'}
    colors = sns.color_palette('colorblind', n_colors=len(data))

    # Plot data
    if args.mode=='Test':
        position = 0

    for runs, label, color in zip(data, labels.get('experiments_name'), colors):
        unique_x_values = set()
        for run in runs:
            for key in run.keys():
                unique_x_values.add(key)
        x_values = sorted(unique_x_values)

        # Plot mean and standard deviation of all runs
        y_values_mean = []
        y_values_std = []
        for x in x_values:
            y_values_mean.append(mean([run.get(x) for run in runs if run.get(x)]))
            y_values_std.append(np.std([run.get(x) for run in runs if run.get(x)]))

        print("%s average result after graphs %f" %(label,y_values_mean[-1]))
        if args.mode =='Train' or args.no_bar_plot:
            x_values.insert(0,0)
            y_values_mean.insert(0,0)
            y_values_std.insert(0,0)
            # Plot std
            ax.fill_between(x_values, np.add(np.array(y_values_mean), np.array(y_values_std)),
                            np.subtract(np.array(y_values_mean), np.array(y_values_std)),
                            alpha=0.3,
                            edgecolor=color, facecolor=color)
            # Plot mean
            plt.plot(x_values, y_values_mean, color=color, linewidth=1.5, label=label)
        else:
            plt.bar(position, y_values_mean, width=0.5, color=color,\
                    label=label, yerr=y_values_std, error_kw=dict(lw=5, capsize=25, capthick=2))
            position += 1

    # Label figure
    ax.legend(loc='lower right', prop={'size': 16})
    if args.mode == 'Train':
        ax.set_xlabel(labels.get('x_label'), **axis_font)
    ax.set_ylabel(labels.get('y_label'), **axis_font)
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(left=0.2)

    # remove grid lines
    ax.grid(False)
    plt.grid(b=False, color='w')
    return fig

def main(args):
    wandb_project = "meta-graph"
    wandb_username = "joey-bose"

    if args.dataset == 'PPI':
        wandb_project = "meta-graph-ppi"
        extract_func = data_to_extract_ppi
    elif args.dataset == 'Reddit':
        wandb_project = "meta-graph-reddit"
        extract_func = data_to_extract_reddit
    elif args.dataset == 'FIRSTMM_DB':
        wandb_project = "meta-graph-firstmmdb"
        extract_func = data_to_extract_firstmmdb
    elif args.dataset=='AMINER':
        wandb_project = "meta-graph-aminer"
        extract_func = data_to_extract_firstmmdb

    labels = extract_func("joeybose",args)
    if args.get_grad_steps:
        data_experiments_auc, data_experiments_ap = get_grad_step_data(args,labels=labels,\
                wandb_project=wandb_project,wandb_username=wandb_username)
        labels['y_label'] = 'AUC'
        fig_auc = my_plot(labels=labels, data=data_experiments_auc)
        labels['y_label'] = 'AP'
        fig_ap = my_plot(labels=labels, data=data_experiments_ap)
        plot_dir = '../plots_datasets/'+ args.dataset + '/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        fig_auc.savefig('../plots_datasets/'+ args.dataset + '/' + \
                        args.name_str+ '_'+ args.file_str + '_' + args.mode +'_grad_AUC.png')
        fig_ap.savefig('../plots_datasets/'+ args.dataset + '/' + args.name_str +\
                       '_' + args.file_str + '_' + args.mode + '_grad_AP.png')
        print("Finished plotting AUC + AP")
    else:
        data_experiments_auc, data_experiments_ap = get_data(args,labels.get('title'), labels.get('x_label'),\
                                    labels.get('y_label'), labels,\
                                    labels.get('experiments_key'),\
                                    wandb_project=wandb_project,\
                                    wandb_username=wandb_username)
        fig_auc = plot(labels=labels, data=data_experiments_auc)
        fig_ap = plot(labels=labels, data=data_experiments_ap)
        if args.local:
            param_str = '_Local_'
        else:
            param_str = '_Global_'

        plot_dir = '../plots_datasets/'+ args.dataset + '/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        fig_auc.savefig('../plots_datasets/'+ args.dataset + '/' + args.file_str + param_str +'_AUC.png')
        fig_ap.savefig('../plots_datasets/'+ args.dataset + '/' + args.file_str + param_str + '_AP.png')
        print("Finished plotting AUC + AP")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_filename', default='plot_source.csv')
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--no_bar_plot", action="store_true", default=False)
    parser.add_argument("--get_grad_steps", action="store_true", default=False)
    parser.add_argument('--mode', type=str, default='Train')
    parser.add_argument('--file_str', type=str, default='')
    parser.add_argument('--name_str', type=str, default='')
    parser.add_argument('--one_maml', type=str, default='')
    parser.add_argument('--two_maml', type=str, default='')
    parser.add_argument('--concat', type=str, default='')
    parser.add_argument('--random_exp', type=str, default='')
    parser.add_argument('--no_finetune', type=str, default='')
    parser.add_argument('--finetune', type=str, default='')
    parser.add_argument('--adamic_adar', type=str, default='')
    parser.add_argument('--mlp', type=str, default='')
    parser.add_argument('--graph_sig', type=str, default='')
    parser.add_argument('--graph_sig_sig', type=str, default='')
    parser.add_argument('--graph_sig_weights', type=str, default='')
    parser.add_argument('--graph_sig_concat', type=str, default='')
    parser.add_argument('--graph_sig_random', type=str, default='')
    parser.add_argument('--dataset', type=str, default='PPI')
    parser.add_argument('--local_block', nargs='+', type=int)
    parser.add_argument('--global_block', nargs='+', type=int)
    parser.add_argument('--get_step', nargs='+', default=5, type=int)
    args = parser.parse_args()

    main(args)

