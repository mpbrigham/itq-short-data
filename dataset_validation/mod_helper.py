import pandas as pd
from natsort import natsorted
import plotly
import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as widgets
from copy import deepcopy

import sys
sys.path.append("../")

import mod_evaluation
import mod_viewer

fig_height = mod_viewer.fig_height
fig_width = mod_viewer.fig_width


def get_model_id(model_id):
    id_split = model_id.split(' + ')
    if len(id_split)>1:
        my_model_id = int(id_split[0])
    else:
        my_model_id = 0
        
    return my_model_id


def load_data(results_path, cache_pre, cache_post, metrics, run_types):

    cache = mod_evaluation.list_cache(results_path)

    cache_lookup = {
        0: '_0_',
        1: '_0_',
        2: '_all_'
    }

    available = {
        item: [
            cache_item
            for cache_item in cache 
            if cache_lookup[item] in cache_item
        ]
        for item in run_types
    }
            
    my_data = {}

    for run_type in available:
        
        info, stats, stats_val = {}, {}, {}

        for item in available[run_type]:

            for metric_id in metrics:

                cache_sig = mod_evaluation.cache_sig_gen(
                    metric_id, 
                    cache_pre=cache_pre, 
                    cache_post=cache_post
                )
        
                if cache_sig in item:
                    
                    run_id = item.split('_')[-1]
                    
                    if run_id not in info:
                        info[run_id] = {}
                        stats[run_id] = {}
                        stats_val[run_id] = {}
                        
                    if metric_id not in info[run_id]:
                        info[run_id][metric_id] = {}
                        stats[run_id][metric_id] = {}
                        stats_val[run_id][metric_id] = {}
                        
                    cache = mod_evaluation.from_cache(
                        item, 
                        results_path
                    )
                    
                    for model_id in cache['info']:
                        
                        my_model_id = get_model_id(model_id)
                        
                        info[run_id][metric_id][my_model_id] = cache['info'][model_id]
                        stats[run_id][metric_id][my_model_id] = cache['stats'][model_id]
                        stats_val[run_id][metric_id][my_model_id] = cache['stats_val'][model_id]

        print('Loaded', run_type)
        
        my_data[run_type] = [
            deepcopy(info), deepcopy(stats), deepcopy(stats_val)
        ]

    return my_data
    

def get_stats_multi(my_data):

    empty = {item: {} for  item in my_data}

    df_multi, df_multi_val = deepcopy(empty), deepcopy(empty)
    df_multi_ca, df_multi_val_ca = deepcopy(empty), deepcopy(empty)

    for run_type in my_data:
        
        info, stats, stats_val = my_data[run_type]
        
        for run_id in info:
                                   
            df_multi[run_type][run_id], df_multi_val[run_type][run_id] = mod_evaluation.get_df_questions(
                info[run_id], stats[run_id], stats_val[run_id],
                ci=False
            )

            df_multi_ca[run_type][run_id], df_multi_val_ca[run_type][run_id] = mod_evaluation.get_df_questions_ca(
                info[run_id], stats[run_id], stats_val[run_id]
            )

    return df_multi, df_multi_val, df_multi_ca, df_multi_val_ca


def get_stats_flat(my_data):

    empty = {item: {} for  item in my_data}

    df_flat, df_flat_val = deepcopy(empty), deepcopy(empty)
    df_flat_ca, df_flat_val_ca = deepcopy(empty), deepcopy(empty)

    for run_type in my_data:

        info, stats, stats_val = my_data[run_type]
        info_flat, stats_flat, stats_val_flat = {}, {}, {}
        
        for run_id in info:
            
            for metric_id in info[run_id]:
                
                if metric_id not in info_flat:
                    info_flat[metric_id], stats_flat[metric_id], stats_val_flat[metric_id] = {}, {}, {}

                for model_id in info[run_id][metric_id]:
                        
                    if model_id not in info_flat:
                        info_flat[metric_id][model_id] = info[run_id][metric_id][model_id]
                        stats_flat[metric_id][model_id] = stats[run_id][metric_id][model_id]
                        stats_val_flat[metric_id][model_id] = stats_val[run_id][metric_id][model_id]
                        
                    else:
                        stats_flat[metric_id][model_id] += stats[run_id][metric_id][model_id]
                        stats_val_flat[metric_id][model_id] += stats_val[run_id][metric_id][model_id]

        df_flat[run_type], df_flat_val[run_type] = mod_evaluation.get_df_questions(
            info_flat, stats_flat, stats_val_flat,
            ci=True
        )

        df_flat_ca[run_type], df_flat_val_ca[run_type] = mod_evaluation.get_df_questions_ca(
            info_flat, stats_flat, stats_val_flat,
        )

    return df_flat, df_flat_val, df_flat_ca, df_flat_val_ca, info_flat


def tab_plot_accuracy_multi(
    df_multi,
    df_multi_val
):

    run_ids = natsorted(df_multi.keys())
    metric_ids = natsorted(df_multi[run_ids[0]])
    
    tab_children = []

    for metric_id in metric_ids:
        
        metric_short = mod_evaluation.sort_params[metric_id]['metric_short']
        
        fig1 = px.scatter(
            None,
            title={'text': 'Mean holdout set accuracy - ' + metric_short, 'x':0.5},
            height=fig_height, width=fig_width
        )
        
        fig2 = px.scatter(
            None,
            title={'text': 'Mean holdout set MSE - ' + metric_short, 'x':0.5},
            height=fig_height, width=fig_width
        )
        
        for run_idx, run_id in enumerate(df_multi):

                if run_idx==0:
                    showlegend = True
                else:
                    showlegend = False

                if ((metric_id not in df_multi[run_id])
                    or (metric_id not in df_multi_val[run_id])):
                    continue

                df_data = df_multi[run_id][metric_id]
                df_data_val = df_multi_val[run_id][metric_id]
                    
                fig1.add_scatter(
                    y=df_data['val_acc_m'],
                    line_color=mod_viewer.rgb_to_rgba(0, 0.2),
                    showlegend=showlegend,
                    name='cv'
                )
                
                fig1.add_scatter(
                    y=df_data_val['val_acc_m'],
                    line_color='rgba(0,0,0,0.2)',
                    showlegend=showlegend,
                    name='hold'
                )
                
                fig2.add_scatter(
                    y=df_data['val_mse_m'],
                    line_color=mod_viewer.rgb_to_rgba(0, 0.2),
                    showlegend=showlegend,
                    name='cv'
                )
                
                fig2.add_scatter(
                    y=df_data_val['val_mse_m'],
                    line_color='rgba(0,0,0,0.2)',
                    showlegend=showlegend,
                    name='hold'
                )

                
        tab_children += [
            widgets.HBox([go.FigureWidget(fig1), go.FigureWidget(fig2)])
        ]

    tab = widgets.Tab()
    tab.children = tab_children

    for metric_idx, metric_id in enumerate(metric_ids):
        metric_short = mod_evaluation.sort_params[metric_id]['metric_short']
        tab.set_title(metric_idx, metric_short)
    tab.style = {'description_width': 'initial'}

    return tab