import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.layers import Input, Dense, Layer, Dropout, Activation
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU

import sklearn as sk
from sklearn import linear_model, preprocessing, pipeline
from sklearn.utils import resample as sk_resample

import pandas as pd

from IPython.display import display
import ipywidgets as widgets
import scipy as sp
import itertools
from natsort import natsorted

import random
import os
import json
import uuid
import json_tricks
import gzip

import plotly
import plotly.graph_objs as go
from plotly import tools

import mod_data
import mod_latent


stats_lookup = { 
    'categorical_accuracy': 'Accuracy train',
    'mean_squared_error': 'MSE train',
    'val_categorical_accuracy': 'Accuracy val',
    'val_mean_squared_error': 'MSE val',
    'drop_n': 'Dropped questions'
}


def quad(my_x):
    
    return np.concatenate([my_x, my_x**2], axis=1)


def model_linear(**kwargs):
    
    m = sk.linear_model.LinearRegression(**kwargs)

    m.name = 'linear'

    return m


def model_linear_ridge(**kwargs):
    
    m = sk.linear_model.Ridge(**kwargs)

    m.name = 'linear_ridge'

    return m


def model_linear_lasso(**kwargs):
    
    m = sk.linear_model.Lasso(**kwargs)

    m.name = 'linear_lasso'

    return m
    

def model_linear_poly(k=2, **kwargs):
    
    m = sk.pipeline.Pipeline([('poly', sk.preprocessing.PolynomialFeatures(degree=k)),
                              ('linear', sk.linear_model.LinearRegression(**kwargs))])

    m.name = 'linear poly' + str(k)

    return m


def model_linear_poly_ridge(k=2, **kwargs):
    
    m = sk.pipeline.Pipeline([('poly', sk.preprocessing.PolynomialFeatures(degree=k)),
                              ('linear', sk.linear_model.Ridge(**kwargs))])

    m.name = 'linear poly ridge' + str(k)

    return m


def model_linear_poly_lasso(k=2, **kwargs):
    
    m = sk.pipeline.Pipeline([('poly', sk.preprocessing.PolynomialFeatures(degree=k)),
                              ('linear', sk.linear_model.Lasso(**kwargs))])

    m.name = 'linear poly lasso' + str(k)
 
    return m


def model_ann(input_shape=32, output_shape=10, h_n=[64, 16], 
              dropout_r=[0.1, 0.1], softmax=True, **kwargs):
    """Define shallow ANN model
    input_shape: input dimension
    output_shape: output dimension
    h_n: list with units per hidden layer
    returns m: shallow ANN Keras model
    """
    
    m = Sequential()
    
    m.add(Layer(input_shape=(input_shape,), name='input'))
    
    for idx_h, n in enumerate(h_n):
        
        name = 'h_'+str(idx_h)

        if idx_h==len(h_n)-1:
            m.add(Dense(n, activation='tanh', name=name))
        else:
            m.add(Dense(n, name=name))
            m.add(PReLU())

        if dropout_r is not None:
            
            if dropout_r[idx_h] is not None:
                
                m.add(Dropout(dropout_r[idx_h], name='dropout_'+str(idx_h)))

    if softmax:    
        m.add(Dense(output_shape, activation='softmax', name='output'))
        loss = 'binary_crossentropy'

    else:
        m.add(Dense(output_shape, name='output'))
        loss = 'mean_squared_error'

    if 'lr' in kwargs:
        sgd = keras.optimizers.Adam(lr=kwargs['lr'])

    else:
        sgd = keras.optimizers.Adam()

    m.compile(optimizer=sgd, loss=loss, metrics=['categorical_accuracy'])

    m.name = 'ann_' + '_'.join([str(input_shape)]
                                + [str(n) for n in h_n]
                                + [str(output_shape)])
    
    return m


def bootstrap_ci(data, alpha=10, n_samples=200, **kwargs):
    
    data_m = np.mean(data)
    boot = sk_resample(data, n_samples=n_samples)
    pct = np.percentile(boot, [alpha/2,100-alpha/2])

    return (2*data_m-pct[1], 2*data_m-pct[0])


def get_stats(y_true, y_pred):

    return {
        'mean_squared_error': sk.metrics.mean_squared_error(y_true, y_pred),
        'categorical_accuracy': sk.metrics.accuracy_score(
            np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)
        ),
        'confusion_matrix': sk.metrics.confusion_matrix(
            np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1),
            labels=range(5)
        )
    }


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


def cross_val(x, y, model_ref, 
             repetitions=10, test_size=0.1,
             epochs=10, batch_size=32, 
             validation_set=None, features_quad=False,
             verbose=0, framework='sklearn', 
             generator=None, **kwargs):

    stats = []

    for n in range(repetitions):

        x_train, x_val, y_train, y_val = sk.model_selection.train_test_split(x, y, test_size=test_size)

        if validation_set:
            x_val, y_val = (validation_set['x'], validation_set['y'])

        if features_quad:
            x_train = quad(x_train.copy())
            x_val = quad(x_val.copy())

        if verbose>0:
            print('repetition:',n,
                '\tx_val\t', x_val.shape[0],
                '\tx_train\t', x_train.shape[0])

        if framework=='keras':

            model = model_ref(x_train.shape[1], 
                            y_train.shape[1],
                            **kwargs)
        
            stats_init_train = model.evaluate(x_train, y_train, 
                                                batch_size=len(y_train), verbose=0)
            stats_init_val = model.evaluate(x_val, y_val, 
                                                    batch_size=len(y_val), verbose=0)
            
            history = model.fit(x_train, y_train,
                                validation_data=(x_val, y_val),
                                epochs=epochs, batch_size=batch_size,
                                verbose=verbose)

            for idx_metric, metric in enumerate(model.metrics_names):

                history.history[metric] = np.concatenate(([stats_init_train[idx_metric]], 
                                                        history.history[metric]))

                history.history['val_'+metric] = np.concatenate(([stats_init_val[idx_metric]], 
                                                                history.history['val_'+metric]))

            stats += [history.history]

        elif framework=='sklearn':

            try:
                model = model_ref(**kwargs).fit(x_train, y_train)

                y_hat = model.predict(x_train)
                y_hat_val = model.predict(x_val)

                my_stats = get_stats(y_train, y_hat)
                my_stats_val = get_stats(y_val, y_hat_val)
                for item in my_stats_val:
                    my_stats['val_'+item] = my_stats_val[item]

                stats += [my_stats]

            except np.linalg.LinAlgError as e:
                print(e)

    return stats


def modelPP_exc_multi(x, n_exc, exclude=None):

    _, y_logits = mod_latent.modelPP(x)

    
    x_cols_ref = mod_data.x_cols_ref
    if exclude is not None:
        x_cols_ref = [
            col for col in x_cols_ref
            if col not in exclude
        ]
        
    stats, info = ({ }, { })
    for cols_exc in itertools.combinations(x_cols_ref, n_exc):

        if exclude is not None:
            cols_exc += tuple(exclude)

        _, y_logits_hat = mod_latent.modelPP_exc(x, cols_exc)

        name = ' + '.join(sorted([item.lower() for item in cols_exc]))
        drop_n = len([
            item for col in cols_exc for item in mod_data.mapping_q_s[col] 
        ])
        
        my_stats = get_stats(y_logits, y_logits_hat)

        stats[name] = {'val_'+item: my_stats[item] for item in my_stats}
        
        info[name] = {
            'drop_n': drop_n,
            'cols_exc': cols_exc
        }
        
    return stats, info


def model_linear_exc_multi(x, y, model_ref, n_exc, 
    validation_set=None, exclude=None, features_quad=True, 
    repetitions=10, test_size=0.1, **kwargs):

    x_cols_ref = mod_data.x_cols_ref
    if exclude is not None:
        x_cols_ref = [
            col for col in x_cols_ref
            if col not in exclude
        ]

    n_trials = int(sp.special.comb(len(x_cols_ref), n_exc))
    progress = widgets.FloatProgress(value=0, min=0, max=n_trials, step=1)
    progress_msg = widgets.Output()
    
    display(widgets.HBox([progress, progress_msg]))
    
    with progress_msg:
        print(
            'evaluating', n_trials, 'models ( CV',
            repetitions, 'repetitions, val size', test_size, ')'
        )
    
    stats, info = ({ }, { })
    for cols_exc in itertools.combinations(x_cols_ref, n_exc):

        if exclude is not None:
            cols_exc += tuple(exclude)

        cols_sel = tuple([idx for idx in range(len(x_cols_ref))
                              if x_cols_ref[idx] not in cols_exc])
        
        name = ' + '.join(sorted([item.lower() for item in cols_exc]))
        drop_n = len([
            item for col in cols_exc for item in mod_data.mapping_q_s[col]
        ])

        my_x = x.copy()[:,cols_sel]
        _, idx_unique = np.unique(my_x, return_index=True, axis=0)

        my_x = my_x[idx_unique]
        my_y = y[idx_unique]

        stats[name] = cross_val(
            my_x, my_y, model_ref, 
            repetitions=repetitions, test_size=test_size,
            validation_set=validation_set,
            features_quad=features_quad, verbose=0, 
            framework='sklearn', **kwargs)

        info[name] = {
            'drop_n': drop_n,
            'cols_exc': cols_exc,
            'x_n': my_x.shape[0]
        }

        progress.value += 1

    return stats, info


def model_linear_exc_question(df, y, model_ref, q_exc=[], 
    validation_set=None, features_quad=True, 
    repetitions=10, test_size=0.1, 
    progress_widgets=None, pi3=False, mapping=None,
    **kwargs):

    my_questions = [
        item for item in mod_data.x_questions_ref
        if item not in q_exc
    ]

    n_trials = len(my_questions) - 1

    if mapping is None:
        mapping = mod_data.mapping_q_s_bis

    if progress_widgets is None:
        progress = widgets.FloatProgress(value=0, min=0, max=n_trials, step=1)
        progress_msg = widgets.Output()
        
        display(widgets.HBox([progress, progress_msg]))

    else:
        progress, progress_msg = progress_widgets
        progress.value = 0
        progress.max = n_trials
        progress_msg.clear_output()

    with progress_msg:
        print(
            'evaluating', n_trials, 'models ( CV',
            repetitions, 'repetitions, val size', test_size, ')'
        )
    
    stats, info = ({ }, { })
    for col_exc in my_questions:
        
        if q_exc:
            name = str(len(set(q_exc))) + ' + ' + col_exc

        else:
            name = col_exc

        cols_sel = [
            item for item in my_questions if item!=col_exc
        ]

        my_df = pd.DataFrame()
        for sumscore in mapping:

            my_cols = [
                item for item in mapping[sumscore]
                    if item in cols_sel
            ]
            if my_cols:
                my_df[sumscore] = np.sum(df[my_cols], axis=1)

        if not pi3:
            cols = [
                item for item in ['pi_org','pi_per','pi_rum']
                    if item in my_df.columns
            ]
            if cols:
                my_df['pi'] = mod_data.pi_linear(my_df)
                my_df = my_df.drop(columns=cols)

        my_cols = list(my_df.columns)

        my_x = np.array(my_df)
        _, idx_unique = np.unique(my_x, return_index=True, axis=0)

        my_x = my_x[idx_unique]
        my_y = y[idx_unique]

        stats[name] = cross_val(
            my_x, my_y, model_ref, 
            repetitions=repetitions, test_size=test_size,
            validation_set=validation_set,
            features_quad=features_quad, verbose=0, 
            framework='sklearn', **kwargs)

        info[name] = {
            'drop_n': len(q_exc)+1,
            'cols_exc': [col_exc]+q_exc,
            'sumscores': my_cols,
            'x_n': my_x.shape[0],
            'x_d': len(my_cols)
        }

        progress.value += 1

    return stats, info


def model_train(
    x_train, y_train, x_val, y_val, model_ref, models, cols_exc,
    features_quad=False, **kwargs
):

    stats_train, stats_val = ({ },{ })

    for name_idx, name in enumerate(models):
        
        cols_sel = tuple([
            idx for idx in range(len(mod_data.x_cols_ref))
                if mod_data.x_cols_ref[idx] not in cols_exc[name_idx]
        ])
            
        my_x_train = x_train.copy()[:,cols_sel]
        my_x_val = x_val.copy()[:,cols_sel]

        _, idx_unique = np.unique(
            np.concatenate((my_x_val, my_x_train), axis=0), 
            return_index=True, axis=0
        )

        idx_unique_val = idx_unique[idx_unique<my_x_val.shape[0]]
        idx_unique_train = idx_unique[idx_unique>=my_x_val.shape[0]] - my_x_val.shape[0]
                
        my_x_val = my_x_val[idx_unique_val]
        my_y_val = y_val[idx_unique_val]

        my_x_train = my_x_train[idx_unique_train]
        my_y_train = y_train[idx_unique_train]
        
        if features_quad:
            my_x_train = quad(my_x_train)
            my_x_val = quad(my_x_val)
            
        model = model_ref(**kwargs).fit(my_x_train, my_y_train)
        
        my_y_hat = model.predict(my_x_train)
        my_y_hat_val = model.predict(my_x_val)

        stats_train[name] = get_stats(my_y_train, my_y_hat)
        stats_val[name] = get_stats(my_y_val, my_y_hat_val)

    return stats_train, stats_val, model


def model_train_questions(
    df_train, y_train, df_val, y_val, model_ref, models, cols_exc,
    features_quad=False, pi3=False, mapping=None,
    **kwargs
    ):

    stats_train, stats_val = ({ },{ })

    for name_idx, name in enumerate(models):

        cols_sel = [
            item for item in mod_data.x_questions_ref
                if item not in cols_exc[name_idx]
        ]        

        if mapping is None:
            mapping = mod_data.mapping_q_s_bis

        my_df_train = pd.DataFrame()
        my_df_val = pd.DataFrame()
        for sumscore in mapping:

            my_cols = [
                item for item in mapping[sumscore]
                    if item in cols_sel
            ]
            my_df_train[sumscore] = np.sum(df_train[my_cols], axis=1)
            my_df_val[sumscore] = np.sum(df_val[my_cols], axis=1)

        if not pi3:
            my_df_train['pi'] = mod_data.pi_linear(my_df_train)
            my_df_val['pi'] = mod_data.pi_linear(my_df_val)
            cols = [
                item for item in ['pi_org','pi_per','pi_rum']
                    if item in my_df_train.columns
            ]            
            if cols:
                my_df_train = my_df_train.drop(columns=cols)
                my_df_val = my_df_val.drop(columns=cols)

        my_x_train = np.array(my_df_train)
        my_x_val = np.array(my_df_val)

        _, idx_unique = np.unique(
            np.concatenate((my_x_val, my_x_train), axis=0), 
            return_index=True, axis=0
        )

        idx_unique_val = idx_unique[idx_unique<my_x_val.shape[0]]
        idx_unique_train = idx_unique[idx_unique>=my_x_val.shape[0]] - my_x_val.shape[0]
                
        my_x_val = my_x_val[idx_unique_val]
        my_y_val = y_val[idx_unique_val]

        my_x_train = my_x_train[idx_unique_train]
        my_y_train = y_train[idx_unique_train]
        
        if features_quad:
            my_x_train = quad(my_x_train)
            my_x_val = quad(my_x_val)
            
        model = model_ref(**kwargs).fit(my_x_train, my_y_train)
        
        my_y_hat = model.predict(my_x_train)
        my_y_hat_val = model.predict(my_x_val)

        stats_train[name] = get_stats(my_y_train, my_y_hat)
        stats_val[name] = get_stats(my_y_val, my_y_hat_val)

    return stats_train, stats_val, model

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