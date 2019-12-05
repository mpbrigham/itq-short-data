import numpy as np
import ipywidgets as widgets
import pandas as pd
from natsort import natsorted
import plotly
import plotly.graph_objects as go
import plotly.express as px

import mod_evaluation

sort_params_short = {
    item: mod_evaluation.sort_params[item]['metric_short']
    for item in mod_evaluation.sort_params
}

fig_height = 450
fig_width = 450

yaxes_range_acc=[0.8, 1]
yaxes_range_mse=[0, 10]
yaxes_range_xent=[0, 0.2]


def tab_plot_metric_class(
    df_questions,
    df_questions_ca,
    questions_info,
    model_type_name=sort_params_short,
    holdout=False,
    yaxes_range_acc=yaxes_range_acc,
    yaxes_range_mse=yaxes_range_mse
):

    tab_name, children = ([], [])

    for model_type in natsorted(questions_info):

        if model_type_name is None:
            name = model_type 
        else:
            name = model_type_name[model_type]

        tab_name += [name]

        if holdout:
            title='Mean holdout set conditional accuracy - ' + name 
        else:
            title='Mean cross-validation conditional accuracy - ' + name 

        fig1 = plot_metric_class(
            df_questions[model_type],
            df_questions_ca[model_type],
            title=title
        )
        
        fig1.update_yaxes(range=yaxes_range_acc)

        if holdout:
            title='Mean holdout set conditional MSE - ' + name 
        else:
            title='Mean cross-validation conditional MSE - ' + name 

        fig2 = plot_metric_class(
            df_questions[model_type],
            df_questions_ca[model_type],
            title=title,
            y_m='val_cond_mse_m',
            y_m_ref='val_mse_m',
        )

        fig2.update_yaxes(range=yaxes_range_mse)

        children += [
            widgets.HBox([go.FigureWidget(fig1), go.FigureWidget(fig2)])
        ]

    tab = widgets.Tab()
    tab.children = children

    for i in range(len(tab_name)):
        tab.set_title(i, tab_name[i])
    tab.style = {'description_width': 'initial'}

    return tab


def tab_plot_metric(
    df_questions, 
    questions_info,
    model_type_name=sort_params_short,
    df_questions_holdout=None,
    yaxes_range_acc=yaxes_range_acc,
    yaxes_range_mse=yaxes_range_mse
):

    tab_name, children = ([], [])

    for model_type in natsorted(questions_info):

        if model_type_name is None:
            name = model_type 
        else:
            name = model_type_name[model_type]
        
        tab_name += [name]

        if df_questions_holdout is None:
            title='Mean cross-validation accuracy - ' + name
        else:
            title='Mean holdout set accuracy - ' + name

        fig1 = plot_metric(
            [[df_questions[model_type], 'cv mean']],
            title=title,
            ci=True
        )
        
        if df_questions_holdout is not None:
            
            fig1.add_scatter(
                x=df_questions_holdout[model_type]['drop_q'],
                y=df_questions_holdout[model_type]['val_acc_m'],
                line_color='rgba(0,0,0,0.6)',
                mode='lines',
                name='holdout'
            )
        
        fig1.update_yaxes(range=yaxes_range_acc)

        if df_questions_holdout is None:
            title='Mean cross-validation MSE - ' + name
        else:
            title='Mean holdout set MSE - ' + name

        fig2 = plot_metric(
            [[df_questions[model_type], 'cv mean']],
            title=title,
            y_m='val_mse_m',
            y_l='val_mse_l', 
            y_u='val_mse_u',   
            ci=True
        )
        
        if df_questions_holdout is not None:
            
            fig2.add_scatter(
                x=df_questions_holdout[model_type]['drop_q'],
                y=df_questions_holdout[model_type]['val_mse_m'],
                line_color='rgba(0,0,0,0.6)',
                mode='lines',
                name='holdout'
            )
        
        fig2.update_yaxes(range=yaxes_range_mse)

        children += [
            widgets.HBox([go.FigureWidget(fig1), go.FigureWidget(fig2)])
        ]

    tab = widgets.Tab()
    tab.children = children

    for i in range(len(tab_name)):
        tab.set_title(i, tab_name[i])
    tab.style = {'description_width': 'initial'}

    return tab


def rgb_to_rgba(color_idx, alpha=1):
    color = plotly.colors.DEFAULT_PLOTLY_COLORS[color_idx]
    return 'rgba('+color[4:-1]+', '+str(alpha)+')'


def plot_metric(
    df_list,
    title='Mean cross-validation accuracy',
    ci=False, 
    y_m='val_acc_m',
    y_l='val_acc_l', 
    y_u='val_acc_u',
    black_trace=None,
    yaxes_range=None
    ):

    for df_idx, (my_df, name) in enumerate(df_list):
        
        if df_idx==0:
            if title is None:
                my_title = None
            else:
                my_title = {'text':title, 'x':0.5}

            fig = px.scatter(
                None,
                title=my_title,
                height=fig_height,
                width=fig_width
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

    if yaxes_range is None:
        y_min = min([item[0][y_m].min() for item in df_list])
        yaxes_range = [min([0.8, y_min]), 1]

    fig.update_yaxes(range=yaxes_range)

    return fig


def plot_metric_class(
    df, 
    df_ca,
    title='Mean cross-validation conditional accuracy',
    y_m='val_cond_acc_m',
    y_m_ref='val_acc_m',
    yaxes_range=None
):

    fig = plot_metric(
        [
            [df_ca[df_ca['class']==k+1], 'class '+str(k+1)]
            for k in range(5)
        ],
        title=title,
        ci=False,
        y_m=y_m,
        yaxes_range=yaxes_range
    )

    fig.add_scatter(
        x=df['drop_q'],
        y=df[y_m_ref],
        line_color='rgba(0,0,0,0.6)',
        mode='lines',
        text=df.index.tolist(),
        name='mean'
    )

    return fig


def plot_metrics(
    df_questions,
    black_trace=5,
    yaxes_range_acc=yaxes_range_acc,
    yaxes_range_mse=yaxes_range_mse,
    ci=True
):
    data = [
        [df_questions[item], sort_params_short[item]]
        for item in sorted(df_questions)
    ]

    fig1 = plot_metric(
        data,
        title='val acc (cross-validation)',
        ci=ci,
        black_trace=black_trace,
        yaxes_range=yaxes_range_acc
    )

    fig2 = plot_metric(
        data,
        title='val xent (cross-validation)',
        ci=ci,
        y_m='val_xent_m',
        y_l='val_xent_l', 
        y_u='val_xent_u',    
        black_trace=black_trace,
        yaxes_range=yaxes_range_xent
    )

    fig3 = plot_metric(
        data,
        title='val mse (cross-validation)',
        ci=ci,
        y_m='val_mse_m',
        y_l='val_mse_l', 
        y_u='val_mse_u',    
        black_trace=black_trace,
        yaxes_range=yaxes_range_mse
    )

    return widgets.HBox([
        go.FigureWidget(fig) for fig in [fig1, fig2, fig3]
        ])