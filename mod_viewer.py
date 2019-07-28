import numpy as np
import ipywidgets as widgets
import pandas as pd
import qgrid
from natsort import natsorted
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mod_evaluation


fig_w, fig_h = (4.5, 3.5)

def get_df_questions(
    questions_stats_best, 
    questions_info_best,
):

    questions_stats_m = mod_evaluation.stats_eval(questions_stats_best)

    questions_stats_ci = mod_evaluation.stats_eval(
        questions_stats_best, 
        fn=lambda a,**kwargs: mod_evaluation.bootstrap_ci(a, alpha=10, **kwargs),
        selected=[
            'categorical_accuracy','val_categorical_accuracy',
            'mean_squared_error','val_mean_squared_error'
        ]
    )


    df = pd.DataFrame([
        [model_id, questions_info_best[model_id]['drop_n'], 17-questions_info_best[model_id]['x_d']]
        + [questions_stats_ci[model_id]['categorical_accuracy'][0], questions_stats_m[model_id]['categorical_accuracy'], questions_stats_ci[model_id]['categorical_accuracy'][1]]
        + [questions_stats_ci[model_id]['val_categorical_accuracy'][0], questions_stats_m[model_id]['val_categorical_accuracy'], questions_stats_ci[model_id]['val_categorical_accuracy'][1]]
        + [questions_stats_ci[model_id]['mean_squared_error'][0], questions_stats_m[model_id]['mean_squared_error'], questions_stats_ci[model_id]['mean_squared_error'][1]]
        + [questions_stats_ci[model_id]['val_mean_squared_error'][0], questions_stats_m[model_id]['val_mean_squared_error'], questions_stats_ci[model_id]['val_mean_squared_error'][1]]    
        for model_id in natsorted(questions_stats_m)
    ])

    df.columns = [
        'model', 'drop_q', 'drop_sum',
        'acc_l', 'acc_m','acc_u',
        'val_acc_l','val_acc_m','val_acc_u',
        'mse_l', 'mse_m', 'mse_u',
        'val_mse_l', 'val_mse_m', 'val_mse_u'
    ]
    df.set_index('model', inplace=True)
    df.sort_values(by=['drop_q'], inplace=True)
    
    return df


def get_df_sumscores(sum_score_stats_multi):

    sum_score_stats_multi_m = {
        key: mod_evaluation.stats_eval(
            sum_score_stats_multi[key][0]
        )
        for key in sum_score_stats_multi
    }
    sum_score_stats_multi_ci = {
        key: mod_evaluation.stats_eval(
            sum_score_stats_multi[key][0], 
            fn=lambda a,**kwargs: mod_evaluation.bootstrap_ci(a, alpha=10, **kwargs),
            selected=[
                'categorical_accuracy','val_categorical_accuracy',
                'mean_squared_error','val_mean_squared_error'
            ]
        )
        for key in sum_score_stats_multi
    }

    df_multi = []
    
    for key in sum_score_stats_multi_m:
        sum_score_info = sum_score_stats_multi[key][1]
        sum_score_stats_m = sum_score_stats_multi_m[key]
        sum_score_stats_ci = sum_score_stats_multi_ci[key]

        df = pd.DataFrame([
            [model_id, sum_score_info[model_id]['drop_n'], int(key)+1]
            + [sum_score_stats_ci[model_id]['categorical_accuracy'][0], sum_score_stats_m[model_id]['categorical_accuracy'], sum_score_stats_ci[model_id]['categorical_accuracy'][1]]
            + [sum_score_stats_ci[model_id]['val_categorical_accuracy'][0], sum_score_stats_m[model_id]['val_categorical_accuracy'], sum_score_stats_ci[model_id]['val_categorical_accuracy'][1]]
            + [sum_score_stats_ci[model_id]['mean_squared_error'][0], sum_score_stats_m[model_id]['mean_squared_error'], sum_score_stats_ci[model_id]['mean_squared_error'][1]]
            + [sum_score_stats_ci[model_id]['val_mean_squared_error'][0], sum_score_stats_m[model_id]['val_mean_squared_error'], sum_score_stats_ci[model_id]['val_mean_squared_error'][1]]    
            for model_id in natsorted(sum_score_stats_m)
        ])

        df.columns = [
            'model', 'drop_q', 'drop_sum',
            'acc_l', 'acc_m','acc_u',
            'val_acc_l','val_acc_m','val_acc_u',
            'mse_l', 'mse_m', 'mse_u',
            'val_mse_l', 'val_mse_m', 'val_mse_u'
        ]
        df.set_index('model', inplace=True)
        df.sort_values(by=['drop_q'], inplace=True)
        df_multi += [df]

    return df_multi


def comparison_table(sum_score_stats_multi, df_questions, df_sumscores_multi):

    tab_name = ['pi + q_n'] + ['pi + s_'+item for item in natsorted(sum_score_stats_multi)]
    children = [
        qgrid.show_grid(df,precision=4) 
        for df in [df_questions]+[df_sumscore for df_sumscore in df_sumscores_multi]
    ]
    tab = widgets.Tab()
    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tab_name[i])

    return tab


def rgb_to_rgba(color_idx, alpha=1):
    color = plotly.colors.DEFAULT_PLOTLY_COLORS[color_idx]
    return 'rgba'+color[3:-1]+', '+str(alpha)+')'


def comparison_plot(
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
        height=1200,
        width=1200,       
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

    fig.show()

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