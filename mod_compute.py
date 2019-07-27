import numpy as np
import scipy as sp
# import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.layers import Input, Dense, Layer, Dropout, Activation
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU

import sklearn as sk
from sklearn import linear_model, preprocessing, pipeline
# from sklearn.utils import resample as sk_resample

import pandas as pd

from IPython.display import display
import ipywidgets as widgets
import itertools
# from natsort import natsorted

# import random
import os
# import json
# import uuid
import json_tricks
import gzip

import mod_data
import mod_latent


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