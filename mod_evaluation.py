import numpy as np
import sklearn as sk
import pandas as pd
from natsort import natsorted
import os
import json_tricks
from copy import deepcopy


sort_params = {

    'mean_squared_error': {
        'metric_sort': 'mean_squared_error',
        'metric_fn': None,
        'metric_sort_max': False,
        'metric_short': 'mse'
    },

    'mean_squared_error_class':  {
        'metric_sort': 'mean_squared_error_class',
        'metric_fn': np.max,
        'metric_sort_max': False,
        'metric_short': 'mse-c'
    },

    'cross_entropy': {
        'metric_sort': 'cross_entropy',
        'metric_fn': None,
        'metric_sort_max': False,
        'metric_short': 'xent'
    },

    'cross_entropy_class': {
        'metric_sort': 'cross_entropy_class',
        'metric_fn': np.max,
        'metric_sort_max': False,
        'metric_short': 'xent-c'
    },

    'categorical_accuracy': {
        'metric_sort': 'categorical_accuracy',
        'metric_fn': None,
        'metric_sort_max': True,
        'metric_short': 'ca'
    },

    'categorical_accuracy_class': {
        'metric_sort': 'categorical_accuracy_class',
        'metric_fn': np.min,
        'metric_sort_max': True,
        'metric_short': 'ca-c'
    },

    'categorical_accuracy_class_prod': {
        'metric_sort': 'categorical_accuracy_class',
        'metric_fn': np.prod,
        'metric_sort_max': True,
        'metric_short': 'ca-p'
    }
}

sort_params_train = sorted(sort_params)

for item in list(sort_params):
    sort_params['val_'+item] = deepcopy(sort_params[item])
    sort_params['val_'+item]['metric_sort'] = 'val_'+sort_params['val_'+item]['metric_sort']

sort_params_val = sorted([
    item
    for item in sort_params 
    if 'val_' in item
])


def cache_sig_gen(
    metric_id, 
    cache_pre='', cache_post=''
):
    cache_sig = ''

    if cache_pre:
        cache_sig += cache_pre

    cache_sig += '_' + metric_id        

    if cache_post:
        cache_sig += '_' + cache_post

    return cache_sig


def list_cache(folder):

    valid = []
    for item in os.listdir(folder):
        compress_name, compress_ext = os.path.splitext(item)
        json_name, json_ext = os.path.splitext(compress_name)
    
        if json_ext+compress_ext=='.json.gz':
            valid += [json_name]
            
    return valid


def to_cache(name, folder, data):
    path = os.path.join(folder, name+'.json.gz')
    with open(path, 'wb') as f:
        json_tricks.dump(data, f, compression=9)

        print('Saved data to',  path)


def from_cache(name, folder):
    path = os.path.join(folder, name+'.json.gz')
    with open(path, 'rb') as f:
        data = json_tricks.load(f, decompression=True)
        
        return data


def bootstrap_ci(data, alpha=5, boot_n=500, **kwargs):
 
    data_m = np.mean(data, axis=0)

    boot_m = np.mean(
        [sk.utils.resample(data) for _ in range(boot_n)],
        axis=0
    )

    ci = np.percentile(boot_m-data_m, [100-alpha/2, alpha/2], axis=0)

    return data_m-ci


def cross_entropy_class(y_true, y_pred):
    
    exp_y_pred = np.exp(y_pred-np.max(y_pred))
    exp_y_true = np.exp(y_true-np.max(y_true))

    softmax_y_pred = exp_y_pred / np.sum(exp_y_pred, axis=1, keepdims=True)
    softmax_y_true = exp_y_true / np.sum(exp_y_true, axis=1, keepdims=True)

    ce = -np.mean(softmax_y_true * np.log(softmax_y_pred), axis=0)

    return ce


def get_stats(y_true, y_pred):

    stats = {
        'mean_squared_error_class': sk.metrics.mean_squared_error(
            y_true, 
            y_pred,
            multioutput='raw_values'
        ), 
        'cross_entropy_class': cross_entropy_class(y_true, y_pred),
        'confusion_matrix': sk.metrics.confusion_matrix(
            np.argmax(y_true, axis=1),
            np.argmax(y_pred, axis=1),
            labels=[0,1,2,3,4]
        )
    }

    return stats


def stats_fill(stats, eps=1e-15):

    my_stats = deepcopy(stats)

    cm = my_stats['confusion_matrix']
    cm_norm  = cm / np.clip(np.sum(cm, axis=1), eps, None)

    cm_val = my_stats['val_confusion_matrix']     
    cm_norm_val  = cm_val / np.clip(np.sum(cm_val, axis=1), eps, None)

    my_stats['mean_squared_error'] = np.mean(my_stats['mean_squared_error_class'])
    my_stats['categorical_accuracy_class'] = np.diag(cm_norm)
    my_stats['categorical_accuracy'] = np.mean(my_stats['categorical_accuracy_class'])
    my_stats['cross_entropy'] = np.mean(my_stats['cross_entropy_class'])

    my_stats['val_mean_squared_error'] = np.mean(my_stats['val_mean_squared_error_class'])
    my_stats['val_categorical_accuracy_class'] = np.diag(cm_norm_val)
    my_stats['val_categorical_accuracy'] = np.mean(my_stats['val_categorical_accuracy_class'])
    my_stats['val_cross_entropy'] = np.mean(my_stats['val_cross_entropy_class'])

    return my_stats


def stats_eval(stats, fn=np.mean, selected=None):

    stats_fn = {}
    for model_id in stats:

        my_stats = [
            stats_fill(item)
            for item in stats[model_id]
        ]

        if selected is None:
            selected = list(my_stats[0])

        stats_fn[model_id] = {}
        for metric in selected:
            stats_fn[model_id][metric] = fn(
                [item[metric] for item in my_stats],
                axis=0
            )

    return stats_fn


def get_df_models(stats, info, ci=True):

    selected = [
        'categorical_accuracy', 'val_categorical_accuracy',
        'cross_entropy', 'val_cross_entropy',
        'mean_squared_error', 'val_mean_squared_error'
    ]

    stats_m = stats_eval(stats, selected=selected)

    if ci:
        stats_ci = stats_eval(
            stats, 
            fn=lambda a,**kwargs: bootstrap_ci(a, alpha=10, **kwargs),
            selected=selected
        )

        df = pd.DataFrame([
            [
                model_id, 
                info[model_id]['drop_n'], 
                16-info[model_id]['x_d'],
                stats_ci[model_id]['val_categorical_accuracy'][0], 
                stats_m[model_id]['val_categorical_accuracy'], 
                stats_ci[model_id]['val_categorical_accuracy'][1],
                stats_ci[model_id]['val_cross_entropy'][0], 
                stats_m[model_id]['val_cross_entropy'], 
                stats_ci[model_id]['val_cross_entropy'][1],
                stats_ci[model_id]['val_mean_squared_error'][0], 
                stats_m[model_id]['val_mean_squared_error'], 
                stats_ci[model_id]['val_mean_squared_error'][1],
                stats_ci[model_id]['categorical_accuracy'][0], 
                stats_m[model_id]['categorical_accuracy'], 
                stats_ci[model_id]['categorical_accuracy'][1],
                stats_ci[model_id]['cross_entropy'][0], 
                stats_m[model_id]['cross_entropy'], 
                stats_ci[model_id]['cross_entropy'][1],
                stats_ci[model_id]['mean_squared_error'][0], 
                stats_m[model_id]['mean_squared_error'], 
                stats_ci[model_id]['mean_squared_error'][1]
            ]
            for model_id in natsorted(stats_m)
        ])

        df.columns = [
            'model', 'drop_q', 'drop_sum',
            'val_acc_l','val_acc_m','val_acc_u',
            'val_xent_l', 'val_xent_m', 'val_xent_u',
            'val_mse_l', 'val_mse_m', 'val_mse_u',
            'acc_l', 'acc_m','acc_u',
            'xent_l', 'xent_m','xent_u',
            'mse_l', 'mse_m', 'mse_u',
        ]
    else:

        df = pd.DataFrame([
            [
                model_id, 
                info[model_id]['drop_n'], 
                16-info[model_id]['x_d'],
                stats_m[model_id]['val_categorical_accuracy'],
                stats_m[model_id]['val_cross_entropy'], 
                stats_m[model_id]['val_mean_squared_error'], 
                stats_m[model_id]['categorical_accuracy'],
                stats_m[model_id]['cross_entropy'], 
                stats_m[model_id]['mean_squared_error']
            ]
            for model_id in natsorted(stats_m)
        ])

        df.columns = [
            'model', 'drop_q', 'drop_sum',
            'val_acc_m', 'val_xent_m', 'val_mse_m',
            'acc_m', 'xent_m', 'mse_m'
        ]

    df.set_index('model', inplace=True)
    df.sort_values(by=['drop_q'], inplace=True)
    
    return df
    
    
def get_df_models_ca(stats, info, eps=1e-8):

    selected = [
        'categorical_accuracy_class', 'val_categorical_accuracy_class',
        'cross_entropy_class', 'val_cross_entropy_class',
        'mean_squared_error_class', 'val_mean_squared_error_class'
    ]

    stats_m = stats_eval(stats, selected=selected)

    df = pd.DataFrame([
            [
                model_id, 
                info[model_id]['drop_n'], 
                16-info[model_id]['x_d'], 
                k+1,
                stats_m[model_id]['val_categorical_accuracy_class'][k],
                stats_m[model_id]['val_cross_entropy_class'][k], 
                stats_m[model_id]['val_mean_squared_error_class'][k],                
                stats_m[model_id]['categorical_accuracy_class'][k],
                stats_m[model_id]['cross_entropy_class'][k], 
                stats_m[model_id]['mean_squared_error_class'][k]                
            ]
            for k in range(5) for model_id in natsorted(stats_m)
    ])

    df.columns = [
        'model', 'drop_q', 'drop_sum', 'class',
        'val_cond_acc_m', 'val_cond_xent_m', 'val_cond_mse_m',
        'cond_acc_m', 'cond_xent_m', 'cond_mse_m'
    ]

    df.sort_values(by=['drop_q', 'class'], inplace=True)
     
    return df


def get_info_stats(results_path, cache_pre, cache_post, metrics):

    available = list_cache(results_path)

    info, stats, stats_val = {}, {}, {}

    for metric_id in metrics:

        cache_sig = cache_sig_gen(
            metric_id, 
            cache_pre=cache_pre, 
            cache_post=cache_post
        )
        
        if cache_sig in available:
            
            cache = from_cache(
                cache_sig, 
                results_path
            )
            
            info[metric_id] = cache['info']
            stats[metric_id] = cache['stats']
            stats_val[metric_id] = cache['stats_val']

            print('Found', metric_id)

    return info, stats, stats_val


def get_df_questions(info, stats, stats_val, ci=True):

    df_questions, df_questions_val = {}, {}

    for metric_id in info:

        df_questions[metric_id] = get_df_models(
            stats[metric_id],
            info[metric_id],
            ci=ci
        )
        
        df_questions_val[metric_id] = get_df_models(
            stats_val[metric_id],
            info[metric_id],
            ci=ci
        )

    return df_questions, df_questions_val


def get_df_questions_ca(info, stats, stats_val):

    df_questions_ca, df_questions_val_ca = {}, {}

    for metric_id in info:

        df_questions_ca[metric_id] = get_df_models_ca(
            stats[metric_id],
            info[metric_id]
        )
        
        df_questions_val_ca[metric_id] = get_df_models_ca(
            stats_val[metric_id],
            info[metric_id]
        )

    return df_questions_ca, df_questions_val_ca