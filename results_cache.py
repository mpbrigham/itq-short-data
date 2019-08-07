import mod_viewer

sources_val = {
    'ca_min': [
        'questions_drop_pi_stats_val_categorical_accuracy_class_min_max',
        'questions_drop_pi_info_val_categorical_accuracy_class_min_max'              
    ],
    'ca_prod': [
        'questions_drop_pi_stats_val_categorical_accuracy_class_prod_max',
        'questions_drop_pi_info_val_categorical_accuracy_class_prod_max'                    
    ],
    'mse': [
        'questions_drop_pi_stats_val_mean_squared_error_min',
        'questions_drop_pi_info_val_mean_squared_error_min'          
    ],
    'mse_max': [
        'questions_drop_pi_stats_val_mean_squared_error_class_max',
        'questions_drop_pi_info_val_mean_squared_error_class_max'
    ],
    'log_loss': [
        'questions_drop_pi_stats_val_log_loss_min',
        'questions_drop_pi_info_val_log_loss_min'           
    ],              
    'log_loss_max': [
        'questions_drop_pi_stats_val_log_loss_class_max',
        'questions_drop_pi_info_val_log_loss_class_max'            
    ]
}


sources_train = {
    'ca_min': [
        'questions_drop_pi_stats_categorical_accuracy_class_min_max',
        'questions_drop_pi_info_categorical_accuracy_class_min_max'              
    ],
    'ca_prod': [
        'questions_drop_pi_stats_categorical_accuracy_class_prod_max',
        'questions_drop_pi_info_categorical_accuracy_class_prod_max'                    
    ],
    'mse': [
        'questions_drop_pi_stats_mean_squared_error_min',
        'questions_drop_pi_info_mean_squared_error_min'          
    ],    
    'mse_max': [
        'questions_drop_pi_stats_mean_squared_error_class_max',
        'questions_drop_pi_info_mean_squared_error_class_max'
    ],
    'log_loss': [
        'questions_drop_pi_stats_log_loss_min',
        'questions_drop_pi_info_log_loss_min'           
    ],             
    'log_loss_max': [
        'questions_drop_pi_stats_log_loss_class_max',
        'questions_drop_pi_info_log_loss_class_max'            
    ]
}


sources_rev = {
    # 'ca_min': [
    #     'questions_drop_pi_stats_val_categorical_accuracy_class_min_max_holdout_train',
    #     'questions_drop_pi_info_val_categorical_accuracy_class_min_max_holdout_train'                    
    # ],
    'ca_prod': [
        'questions_drop_pi_stats_val_categorical_accuracy_class_prod_max_holdout_train',
        'questions_drop_pi_info_val_categorical_accuracy_class_prod_max_holdout_train'                    
    ],
    # 'mse': [
    #     'questions_drop_pi_stats_val_mean_squared_error_min_holdout_train',
    #     'questions_drop_pi_info_val_mean_squared_error_min_holdout_train'                    
    # ]    
}


model_type_name = {
    'ca_min': 'conditional accuracy min',
    'ca_prod': 'conditional accuracy product',
    'log_loss_max': 'conditional cross-entropy max',
    'log_loss': 'cross-entropy',
    'mse': 'mean square error',
    'mse_max': 'conditional mean square error max'
}


def get_questions(results_folder, sources=sources_val):
    
    questions_stats, questions_info = ({}, {})

    for name in sources:

        questions_stats[name], questions_info[name] = mod_viewer.from_cache_multiple(
            sources[name], 
            results_folder
        )
        print('Loaded', name)

    return questions_stats, questions_info


def get_df_questions(questions_stats, questions_info):

    df_questions, df_questions_ca = ({}, {})

    for name in questions_stats:

        df_questions[name] = mod_viewer.get_df_questions(
            questions_stats[name],
            questions_info[name]
        )

        df_questions_ca[name] = mod_viewer.get_df_questions_conditional_accuracy(
            questions_stats[name],
            questions_info[name]
        )

    return df_questions, df_questions_ca
