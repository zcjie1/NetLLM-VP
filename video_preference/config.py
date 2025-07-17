import os
"""
This file contains the configuration information
"""


class Config:
    dataset_dir = './dataset'
    dataset_name = 'total_llm.csv'
    split_type = None
    his_window = 3
    video_len = 10
    fut_window = 3
    dataset_image_features = './video_feature'
    plm_types = ['gpt2', 'llama', 'llava', 't5-lm', 'opt', 'mistral']
    plm_sizes = ['xxs', 'xs', 'small', 'base', 'large', 'xl', 'xxl']  # note that the actual size of plm is dependent on the type of plm. 
                                                         # for example, for llama, 'base' is 7b, while for gpt2, 'base' is 340M. you can specify it yourself.
    _base_dir = '' if 'video_prefernce' in os.getcwd() else 'video_prefernce/'
    plms_finetuned_dir = _base_dir + 'data/ft_plms'
    models_dir =  _base_dir + 'data/models'
    results_dir = _base_dir + 'data/results'
    plms_dir = _base_dir + ('../../downloaded_plms' if 'video_prefernce' in _base_dir else '../downloaded_plms')
    
    # time-series model default settings
    default_bs = 1  # batch size
    default_grad_accum_step = 32  # gradient accumulation steps
    default_lr = 2e-4  # learning rate
    default_weight_decay = 1e-5  # weight decay

    default_epochs = 40  # training epochs

cfg = Config()
