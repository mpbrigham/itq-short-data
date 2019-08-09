import numpy as np
import ipywidgets as widgets
import pandas as pd
import qgrid
from natsort import natsorted
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mod_evaluation
import results_cache

fig_w, fig_h = (4.5, 3.5)


def from_cache_multiple(names, folder):
    return [mod_evaluation.from_cache(item, folder) for item in names]        


def get_df_sumscores(stats_multi):

    stats_multi_m = {
        key: mod_evaluation.stats_eval(
            stats_multi[key][0]
        )
        for key in stats_multi
    }
    stats_multi_ci = {
        key: mod_evaluation.stats_eval(
            stats_multi[key][0], 
            fn=lambda a,**kwargs: mod_evaluation.bootstrap_ci(a, alpha=10, **kwargs),
            selected=[
                'categorical_accuracy','val_categorical_accuracy',
                'mean_squared_error','val_mean_squared_error'
            ]
        )
        for key in stats_multi
    }

    df_multi = []
    
    for key in stats_multi_m:
        info = stats_multi[key][1]
        stats_m = stats_multi_m[key]
        stats_ci = stats_multi_ci[key]

        df = pd.DataFrame([
            [
                model_id, 
                info[model_id]['drop_n'],
                int(key)+1,
                stats_ci[model_id]['val_categorical_accuracy'][0], 
                stats_m[model_id]['val_categorical_accuracy'], 
                stats_ci[model_id]['val_categorical_accuracy'][1],
                stats_ci[model_id]['val_mean_squared_error'][0], 
                stats_m[model_id]['val_mean_squared_error'], 
                stats_ci[model_id]['val_mean_squared_error'][1],
                stats_ci[model_id]['categorical_accuracy'][0], 
                stats_m[model_id]['categorical_accuracy'], 
                stats_ci[model_id]['categorical_accuracy'][1],
                stats_ci[model_id]['mean_squared_error'][0], 
                stats_m[model_id]['mean_squared_error'], 
                stats_ci[model_id]['mean_squared_error'][1]
            ]
            for model_id in natsorted(stats_m)
        ])

        df.columns = [
            'model', 'drop_q', 'drop_sum',
            'val_acc_l','val_acc_m','val_acc_u',
            'val_mse_l', 'val_mse_m', 'val_mse_u',
            'acc_l', 'acc_m','acc_u',
            'mse_l', 'mse_m', 'mse_u'            
        ]
        df.set_index('model', inplace=True)
        df.sort_values(by=['drop_q'], inplace=True)
        df_multi += [df]

    return df_multi


def get_df_questions(
    stats_best, 
    info_best,
    ci=True
):

    stats_m = mod_evaluation.stats_eval(stats_best)

    if ci:
        stats_ci = mod_evaluation.stats_eval(
            stats_best, 
            fn=lambda a,**kwargs: mod_evaluation.bootstrap_ci(a, alpha=10, **kwargs),
            selected=[
                'categorical_accuracy', 'val_categorical_accuracy',
                'mean_squared_error', 'val_mean_squared_error'
            ]
        )

        df = pd.DataFrame([
            [
                model_id, 
                info_best[model_id]['drop_n'], 
                17-info_best[model_id]['x_d'],
                stats_ci[model_id]['val_categorical_accuracy'][0], 
                stats_m[model_id]['val_categorical_accuracy'], 
                stats_ci[model_id]['val_categorical_accuracy'][1],
                stats_ci[model_id]['val_mean_squared_error'][0], 
                stats_m[model_id]['val_mean_squared_error'], 
                stats_ci[model_id]['val_mean_squared_error'][1],
                stats_ci[model_id]['categorical_accuracy'][0], 
                stats_m[model_id]['categorical_accuracy'], 
                stats_ci[model_id]['categorical_accuracy'][1],
                stats_ci[model_id]['mean_squared_error'][0], 
                stats_m[model_id]['mean_squared_error'], 
                stats_ci[model_id]['mean_squared_error'][1]
            ]
            for model_id in natsorted(stats_m)
        ])

        df.columns = [
            'model', 'drop_q', 'drop_sum',
            'val_acc_l','val_acc_m','val_acc_u',
            'val_mse_l', 'val_mse_m', 'val_mse_u',
            'acc_l', 'acc_m','acc_u',  
            'mse_l', 'mse_m', 'mse_u',
        ]
    else:

        df = pd.DataFrame([
            [
                model_id, 
                info_best[model_id]['drop_n'], 
                17-info_best[model_id]['x_d'],
                stats_m[model_id]['val_categorical_accuracy'], 
                stats_m[model_id]['val_mean_squared_error'], 
                stats_m[model_id]['categorical_accuracy'], 
                stats_m[model_id]['mean_squared_error']
            ]
            for model_id in natsorted(stats_m)
        ])

        df.columns = [
            'model', 'drop_q', 'drop_sum',
            'val_acc_m', 'val_mse_m',
            'acc_m', 'mse_m'
        ]

    df.set_index('model', inplace=True)
    df.sort_values(by=['drop_q'], inplace=True)
    
    return df
    
    
def get_df_questions_conditional_accuracy(
    stats_best, 
    info_best,
    ci=True
):

    stats_m = mod_evaluation.stats_eval(stats_best)

    if ci:
        stats_ci = mod_evaluation.stats_eval(
            stats_best, 
            fn=lambda a,**kwargs: mod_evaluation.bootstrap_ci(a, alpha=10, **kwargs),
            selected=[
                'categorical_accuracy_class', 'val_categorical_accuracy_class'
            ]
        )
    
        df = pd.DataFrame([
                [
                    model_id, 
                    info_best[model_id]['drop_n'], 
                    17-info_best[model_id]['x_d'], 
                    k+1,
                    stats_ci[model_id]['val_categorical_accuracy_class'][0][k],
                    stats_m[model_id]['val_categorical_accuracy_class'][k],
                    stats_ci[model_id]['val_categorical_accuracy_class'][1][k],
                    stats_ci[model_id]['categorical_accuracy_class'][0][k],
                    stats_m[model_id]['categorical_accuracy_class'][k],
                    stats_ci[model_id]['categorical_accuracy_class'][1][k]
                ]
                for k in range(5) for model_id in natsorted(stats_m)
        ])
        
        df.columns = [
            'model', 'drop_q', 'drop_sum', 'class',
            'val_cond_acc_l', 'val_cond_acc_m','val_cond_acc_u',
            'cond_acc_l', 'cond_acc_m','cond_acc_u'
        ]

    else:
        df = pd.DataFrame([
                [
                    model_id, 
                    info_best[model_id]['drop_n'], 
                    17-info_best[model_id]['x_d'], 
                    k+1,
                    stats_m[model_id]['val_categorical_accuracy_class'][k],
                    stats_m[model_id]['categorical_accuracy_class'][k]
                ]
                for k in range(5) for model_id in natsorted(stats_m)
        ])

        df.columns = [
            'model', 'drop_q', 'drop_sum', 'class',
            'val_cond_acc_m', 'cond_acc_m'
        ]

    df.sort_values(by=['drop_q', 'class'], inplace=True)
     
    return df


def table_accuracy_questions_sum_score(
    sum_score_stats_multi, 
    df_questions, 
    df_sumscores_multi
):

    tab_name = ['pi + q_n'] + [
        'pi + s_'+item 
        for item in natsorted(sum_score_stats_multi)
        ]
    children = [
        qgrid.show_grid(df,precision=4) 
        for df in [df_questions]+[df_sumscore 
        for df_sumscore in df_sumscores_multi]
    ]
    tab = widgets.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tab_name[i])
    tab.style = {'description_width': 'initial'}

    return tab


def table_conditional_accuracy(df):

    tab_name = ['class '+str(k+1) for k in range(5)]
    children = [
        qgrid.show_grid(
            df[[
                'model', 'drop_q', 'drop_sum', 'class',
                'cond_acc_l', 'cond_acc_m', 'cond_acc_u',
                'val_cond_acc_l', 'val_cond_acc_m', 'val_cond_acc_u'
            ]][df['class']==k+1],
            precision=4 
        )
        for k in range(5)
    ]

    tab = widgets.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tab_name[i])
    tab.style = {'description_width': 'initial'}

    return tab


def table_accuracy_holdout(stats_holdout, questions_info):

    tab_name, children = ([], [])

    for model_type in natsorted(stats_holdout):
        
        tab_name += [model_type]
        
        df = pd.DataFrame([
            [
                model_id, 
                questions_info[model_type][model_id]['drop_n'],
                stats_holdout[model_type][model_id]['val_categorical_accuracy'],      
                stats_holdout[model_type][model_id]['val_mean_squared_error'], 
                stats_holdout[model_type][model_id]['categorical_accuracy'],
                stats_holdout[model_type][model_id]['mean_squared_error'], 
            ]
            for model_id in natsorted(stats_holdout[model_type])
        ])
        
        df.columns = [
            'model', 'drop_q', 
            'val_acc_m', 'val_mse_m',
            'acc_m', 'mse_m'
        ]
            
        df.set_index('model', inplace=True)
        
        children += [qgrid.show_grid(df, precision=4)]

    tab = widgets.Tab()
    tab.children = children
    for i in range(len(tab_name)):
        tab.set_title(i, tab_name[i])
    tab.style = {'description_width': 'initial'}
        
    return tab


def tab_plot_conditional_accuracy(
    df_questions,
    df_questions_ca,
    questions_info,
    model_type_name=None,
    title=None
):

    tab_name, children = ([], [])

    for model_type in natsorted(questions_info):

        if model_type_name is None:
            name = model_type 
        else:
            name = model_type_name[model_type]

        if title is None:
            title='Conditional accuracy - validation - ' + name

        tab_name += [model_type]

        fig = plot_conditional_accuracy(
            df_questions[model_type],
            df_questions_ca[model_type],
            title=title 
        )
               
        fig.update_yaxes(range=[0.8, 1])
        
        children += [go.FigureWidget(fig)]

    tab = widgets.Tab()
    tab.children = children

    for i in range(len(tab_name)):
        tab.set_title(i, tab_name[i])
    tab.style = {'description_width': 'initial'}

    return tab


def tab_plot_descent_methods_validation(
    df_questions, 
    questions_info,
    model_type_name=None,
    stats_holdout=None
):

    tab_name, children = ([], [])

    for model_type in natsorted(questions_info):

        if model_type_name is None:
            name = model_type 
        else:
            name = model_type_name[model_type]
        
        tab_name += [model_type]

        fig = plot_simple(
            [[df_questions[model_type], 'cv mean']],
            title='Holdout set accuracy - '+name,
            ci=True
        )
        
        if stats_holdout is not None:

            models = natsorted(stats_holdout[model_type])
            
            x = [
                questions_info[model_type][model_id]['drop_n'] 
                for model_id in models
            ]
            
            y = [
                stats_holdout[model_type][model_id]['val_categorical_accuracy']  
                for model_id in models
            ]
            
            fig.add_scatter(
                x=x,
                y=y,
                line_color='rgba(0,0,0,0.6)',
                mode='lines',
                text=models,
                name='holdout'
            )
        
        fig.update_yaxes(range=[0.8, 1])
        
        children += [go.FigureWidget(fig)]

    tab = widgets.Tab()
    tab.children = children

    for i in range(len(tab_name)):
        tab.set_title(i, tab_name[i])
    tab.style = {'description_width': 'initial'}

    return tab


def rgb_to_rgba(color_idx, alpha=1):
    color = plotly.colors.DEFAULT_PLOTLY_COLORS[color_idx]
    return 'rgba('+color[4:-1]+', '+str(alpha)+')'


def plot_accuracy_mse(
    sum_score_stats_multi,
    df_questions,
    df_sumscores_multi
):

    params = [
        ['val_acc_m', 'val_acc_u', 'val_acc_l', 'Accuracy - validation'],
        ['val_mse_m', 'val_mse_u', 'val_mse_l', 'MSE - validation']
    ]
    
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=[item[-1] for item in params],
        vertical_spacing=0.1
    )

    fig.update_layout(
        height=600,
        width=800    
    )

    for plot_idx, (y_name, y_u_name, y_l_name, _) in enumerate(params):  
        
        if plot_idx==0:
            showlegend=True
        else:
            showlegend=False

        for trace_idx, (name, data) in enumerate(zip(
            ['pi+q_n']+ ['pi + s_'+item for item in natsorted(sum_score_stats_multi)],
            [df_questions]+[df_sumscore for df_sumscore in df_sumscores_multi]
        )):

            x = data['drop_q'].tolist()
            y = data[y_name].tolist()
            y_u = data[y_u_name].tolist()
            y_l = data[y_l_name].tolist()
            text = data.index.values.tolist()

            fig.add_trace(
                go.Scatter(
                    x=x+x[::-1],
                    y=y_u+y_l[::-1],
                    fill='toself',
                    fillcolor=rgb_to_rgba(trace_idx, 0.2),
                    line_color='rgba(255,255,255,0)',
                    name=name+' ci',
                    showlegend=showlegend,
                    legendgroup='m'+str(trace_idx)
                ),
                row=plot_idx+1,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    line_color=rgb_to_rgba(trace_idx, 0.6),
                    mode='lines',
                    text=text,
                    name=name,
                    showlegend=showlegend,
                    legendgroup='ci'+str(trace_idx)
                ),
                row=plot_idx+1,
                col=1
            )

    return fig


def plot_simple(
    df_list,
    title=None,
    ci=False, 
    y_m='val_acc_m',
    y_l='val_acc_l', 
    y_u='val_acc_u',
    black_trace=None,
    yaxes_range=None
    ):

    for df_idx, (my_df, name) in enumerate(df_list):

        if df_idx==0:
            fig = px.scatter(
                my_df,
                title={'text':title, 'x':0.5},
                height=600,
                width=800
            )

        if black_trace is not None and black_trace==df_idx:
            fillcolor='rgba(0,0,0,0.2)'
            line_color='rgba(0,0,0,0.6)'
        else:
            fillcolor=rgb_to_rgba(df_idx, 0.2)
            line_color=rgb_to_rgba(df_idx, 0.6)

        x = my_df['drop_q'].tolist()
        y = my_df[y_m].tolist()

        if ci:

            data_l = my_df[y_l].tolist() 
            data_u = my_df[y_u].tolist()

            fig.add_scatter(
                x=x+x[::-1],
                y=data_u+data_l[::-1],
                fill='toself',
                fillcolor=fillcolor,
                line_color='rgba(255,255,255,0)',
                name=name+' ci'
            )

        fig.add_scatter(
            x=x, y=y,
            line_color=line_color,
            mode='lines',
            text=my_df.index.tolist(),
            name=name
        )

    if yaxes_range is not None:
        fig.update_yaxes(range=yaxes_range)

    return fig


def plot_conditional_accuracy(
    df, 
    df_ca,
    title='Conditional accuracy - validation'
):

    fig = plot_simple(
        [
            [df_ca[df_ca['class']==k+1], 'class '+str(k+1)]
            for k in range(5)
        ],
        title=title,
        ci=False,
        y_m='val_cond_acc_m'
    )

    fig.add_scatter(
        x=df['drop_q'],
        y=df['val_acc_m'],
        line_color='rgba(0,0,0,0.6)',
        mode='lines',
        text=df.index.tolist(),
        name='mean'
    )

    return fig


def comparison_plot_old(
    questions_stats_best,
    questions_info_best,
    sum_score_stats_multi
):

    plot_args = [
        ['val_categorical_accuracy', 'Accuracy - validation', [0.7,1], [20,150]],
        ['val_categorical_accuracy', 'Accuracy - validation (zoom)', [0.8,1], [50,100]],
        ['val_mean_squared_error', 'MSE - validation', [0,10], [20,150]],
        ['val_mean_squared_error', 'MSE - validation (zoom)', [0,10], [50,100]],
    ]

    fig, ax = plt.subplots(2, 2, figsize=(2.5*fig_w, 2.5*fig_h))

    for my_idx, (my_metric, my_title, my_ylim, my_xlim) in enumerate(plot_args):
        plt.sca(ax.flatten()[my_idx])
        mod_evaluation.plot_stats_evolution(
            questions_stats_best, questions_info_best, my_metric, 
            models=natsorted(questions_stats_best), recycle=True, 
            ci=None, label='pi + q_n'
        )
        
        for n in sorted(sum_score_stats_multi):
            stats_best, info_best = sum_score_stats_multi[n]
            models_list = list(stats_best)

            top_idx = np.argsort([info_best[key]['drop_n'] for key in models_list])
            models_sorted_n = [models_list[idx] for idx in top_idx]
            label = 'pi + s_' + str(n)

            mod_evaluation.plot_stats_evolution(
                stats_best, info_best, my_metric,
                models=models_sorted_n, color='C'+str(n),
                recycle=True, ci=None, label=label, alpha_mean=0.6
            )
            
        plt.gca().set_title(my_title)
        plt.ylim(my_ylim)
        plt.xlim(my_xlim)
        if my_idx==0:
            plt.legend()
        plt.grid(axis='both')
        
    plt.tight_layout()
    fig.canvas.draw()
    plt.show()       

    return fig     