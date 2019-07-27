import numpy as np

# import pandas as pd

import sklearn as sk
from sklearn.utils import resample as sk_resample

# from IPython.display import display
# import ipywidgets as widgets

from natsort import natsorted

import gzip
import os
import json_tricks

import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go
from plotly import tools


stats_lookup = { 
    'categorical_accuracy': 'Accuracy train',
    'mean_squared_error': 'MSE train',
    'val_categorical_accuracy': 'Accuracy val',
    'val_mean_squared_error': 'MSE val',
    'drop_n': 'Dropped questions'
}

def bootstrap_ci(data, alpha=10, n_samples=200, **kwargs):
    
    data_m = np.mean(data)
    boot = sk.utils.resample(data, n_samples=n_samples)
    pct = np.percentile(boot, [alpha/2,100-alpha/2])

    return (2*data_m-pct[1], 2*data_m-pct[0])


def stats_eval(my_stats, fn=np.mean, selected=None):

    my_stats_fn = { }
    for key in my_stats:
        my_stats_fn[key] = { }
        if isinstance(my_stats[key], list):
            for metric in my_stats[key][0]:
                if selected is not None and metric not in selected:
                    continue
                my_stats_fn[key][metric] = fn(
                    [item[metric] for item in my_stats[key]],
                    axis=0
                )
        else:
            my_stats_fn[key] = my_stats[key]

    return my_stats_fn


def to_cache(name, folder, data):

    with gzip.GzipFile(os.path.join(folder, name+'.json.gz'), 'w') as f:
        f.write(json_tricks.dumps(data).encode('utf-8'))

def from_cache(name, folder):
    data = None
    with gzip.GzipFile(os.path.join(folder, name+'.json.gz'), 'r') as f:
        data = json_tricks.loads(f.read().decode('utf-8'))
        
    return data


def top_stats(
    my_stats, my_info,
    n=10, revsort=True, info_sort='drop_n', 
    metric_sort=None, metric_sort_fn=np.mean,
    metric_filter=None, metric_filter_fn=np.mean,
    ):

    models_list = list(my_stats)

    my_stats_sort = stats_eval(my_stats, fn=metric_sort_fn)
    my_stats_filter = stats_eval(my_stats, fn=metric_filter_fn)

    if metric_sort:
        top_idx = np.argsort([
            my_stats_sort[item][metric_sort] for item in models_list
        ])
    else:
        top_idx = np.argsort([
            my_info[item][info_sort] for item in models_list
        ])

    if revsort:
        top_idx = top_idx[::-1]

    selected = [models_list[idx] for idx in top_idx]

    if metric_filter:
        metric, metric_th = metric_filter
        selected = [
            item for item in selected
                if my_stats_filter[item][metric]>=metric_th
        ]

    if n is not None:
        selected = selected[:n]

    return selected


def print_top(my_stats, my_info,
    n=10, revsort=True, info_sort='drop_n', 
    metric_sort=None, metric_sort_fn=np.mean, 
    metric_filter=None, metric_filter_fn=np.mean,
    metric_show='val_categorical_accuracy',
    print_selected=False
    ):
    
    selected = top_stats(
        my_stats, my_info,
        n=n, revsort=revsort, info_sort=info_sort,  
        metric_sort=metric_sort, metric_sort_fn=metric_sort_fn,
        metric_filter=metric_filter, metric_filter_fn=metric_filter_fn
    )

    if revsort:
        title_prefix = 'top'
        
    else:
        title_prefix = 'bottom'
    
    if selected:

        my_stats_mean = stats_eval({key: my_stats[key] for key in selected})
        my_stats_min = stats_eval({key: my_stats[key] for key in selected}, fn=np.min)
        my_stats_max = stats_eval({key: my_stats[key] for key in selected}, fn=np.max)

        stats_keys_len = max([len(key) for key in selected])
        str_format = '{:'+str(stats_keys_len)+'s}'
        str_metric = ' '*(stats_keys_len-len(title_prefix)) + metric_show

        print('[', title_prefix, str_metric, '   '+info_sort,']')

        for key in selected:
            print(
                str_format.format(key), 
                '  {:.4f}'.format(my_stats_mean[key][metric_show]),
                ' ({:.4f} - {:.4f})'.format(
                    my_stats_min[key][metric_show],
                    my_stats_max[key][metric_show]), 
                ' ', my_info[key][info_sort]
            )
              
        print()
        if print_selected:
            selected_names = [item.split()[-1] for item in selected]
            print("'" + "', '".join(selected_names))


def plot_stats(
    my_stats, my_info, 
    info_sort='drop_n', metric='val_categorical_accuracy',
    plot_order=None,
    title=None, xlim=None, ylim=None, 
    fig_w=4.5, fig_h=3.5, recycle=False
):
    if plot_order is None:
        plot_order = my_stats.keys()

    if recycle:
        fig = plt.gcf()
    else:
        fig = plt.figure(figsize=(fig_w, fig_h))

    colors = plt.cm.get_cmap('tab20')(np.linspace(0,1,20))

    for key_idx, key in enumerate(plot_order):

        x = my_info[key][info_sort]
        metric_m  = np.mean([item[metric] for item in my_stats[key]])

        for item in my_stats[key]:
            plt.plot(
                x, item[metric], 
                color=colors[key_idx], marker='_'
            )
        
        plt.plot(
            x, metric_m, 
            color=colors[key_idx], marker='_', 
            markeredgewidth=3, markersize=10, label=key
        )

    if title is not None:
        plt.title(stats_lookup[metric])

    plt.ylabel(stats_lookup[metric])
    plt.xlabel(stats_lookup[info_sort])

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    plt.tight_layout()
    fig.canvas.draw()
    plt.show()


def plot_latent(
    x_latent, 
    y=None, title='Latent space',
    fig_w=4.5, fig_h=3.5, alpha=0.7, 
    recycle=False
):
    
    fig_n = len(x_latent)
    
    if not recycle:
        fig, ax = plt.subplots(1, fig_n, figsize=(fig_n*fig_w, fig_h))
        
    else:
        fig = plt.gcf()
        ax = fig.axes           

    for my_x_idx, my_x in enumerate(x_latent):
        
        if type(ax).__name__=='AxesSubplot':
            plt.sca(ax)
            
        else:
            plt.sca(ax[my_x_idx])

        if y is None:
            plt.scatter(my_x[:,0], my_x[:,1],
                        c='gray', s=30,
                        alpha=alpha, edgecolors='none'
            )

        else:

            my_y = y[my_x_idx]

            for cluster in set(my_y.flatten()):
                plt.scatter(my_x[(my_y.flatten()==cluster)[0],0],
                            my_x[(my_y.flatten()==cluster)[0],1],
                            c=['C'+str(cluster-1)],
                            s=30,
                            alpha=alpha,
                            edgecolors='none',
                            label=cluster) 
                
            plt.scatter(my_x[:,0], my_x[:,1],
                        c=['C'+str(cluster-1) for cluster in my_y.flatten()],
                        s=30,
                        alpha=alpha,
                        edgecolors='none') 
     
            plt.legend()
            
        if type(title) in [list,tuple]:
            plt.title(title[my_x_idx])
        else:
            plt.title(title)    
            
        plt.xticks([])
        plt.yticks([])

    if not recycle:  
        plt.tight_layout()
        fig.canvas.draw()

    return


def plot_stats_evolution(
    my_stats_best, my_info_best, metric,
    fig_w=4.5, fig_h=3.5,
    alpha_mean=0.8, alpha_ci=0.6, alpha_scatter=0.1,
    color=None, label=None,
    models=None, pct_alpha=10, ci='pct',
    recycle=False
):
    if color is None:
        color = 'C0'

    if models is None:
        models = natsorted(my_stats_best)
        
    drop_n = [my_info_best[item]['drop_n'] for item in models]

    stats_mean = stats_eval(my_stats_best, selected=[metric])
    metric_mean = [stats_mean[model][metric] for model in models]
    
    if ci=='pct':

        stats_pct = stats_eval(
            my_stats_best,
            selected=[metric],
            fn=lambda a,**kwargs: (
                np.percentile(a, [pct_alpha/2,100-pct_alpha/2], **kwargs)
            )
        )
        
        metric_l = [stats_pct[model][metric][0] for model in models]
        metric_u = [stats_pct[model][metric][1] for model in models]

    elif ci=='bootstrap':

        stats_bs = stats_eval(
            my_stats_best, 
            selected=[metric],
            fn=lambda a,**kwargs: bootstrap_ci(a, alpha=pct_alpha, **kwargs)
        )

        metric_l = [stats_bs[model][metric][0] for model in models]
        metric_u = [stats_bs[model][metric][1] for model in models]

    if not recycle:
        fig = plt.figure(figsize=(fig_w, fig_h))
        
    else:
        fig = plt.gcf()

    if ci in ['pct', 'bootstrap']:

        plt.fill_between(drop_n, metric_l, metric_u, color=color, linewidth=0, alpha=alpha_ci)  

    elif ci=='samples':

        for model, n in zip(models, drop_n):
            
            data = np.array([item[metric] for item in my_stats_best[model]])
            
            plt.scatter(
                [n]*data.shape[0], data, s=100, marker='.', color='k',
                alpha=alpha_scatter, edgecolors='none'
            )

    plt.plot(
        drop_n, metric_mean, 
        color=color, linewidth=2, 
        alpha=alpha_mean, label=label
    )

    if not recycle:
        
        plt.tight_layout()
        fig.canvas.draw()
        plt.show()