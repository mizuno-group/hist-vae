# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

arguments
※scratch先のrepoにはない

@author: tadahaya
"""

import os
import torch
import numpy as np
import argparse
import random
import re
import yaml
import shutil
import warnings
from datetime import datetime

class NameSpace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z]_-", key)
            if isinstance(value, dict):
                self.__dict__[key] = NameSpace(value) # keyがstrであることを担保
            else:
                self.__dict__[key] = value
        
    def __getattr__(self, attribute):
        raise AttributeError(
            f"Can not find {attribute} in namespace. Write {attribute} in your config file (xxx.yaml)!"
            )
        # attributeを直接引けないようにして動線をyamlに絞る


def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    # なくす
    parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    parser.add_argument('--log_dir', type=str, default=os.getenv('LOG'))
    parser.add_argument('--ckpt_dir', type=str, default=os.getenv('CHECKPOINT'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in NameSpace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value # argsの中身取り出す
    
    if args.debug:
        if args.train:
            args.train.batch_size = 2
            args.train.num_epoch = 1
            args.train.stop_at_epoch = 1
        if args.eval:
            args.eval.batch_size = 2
            args.eval.num_epochs = 1
        args.dataset.num_workers = 0


    assert not None in [args.log_dir, args.dat_dir, args.ckpt_dir, args.name]       

    args.log_dir = os.path.join(args.log_dir, 'in-progress_'+datetime.now().strftime('%m%d%H%M%S_')+args.name)

    os.makedirs(args.log_dir, exist_ok=False)
    print(f'creating file {args.log_dir}')

    shutil.copy2(args.config_file, args.log_dir)

    vars(args)['aug_kwargs'] = {
        'name':args.model.name,
        'image_size':args.dataset.image_size # 変更予定
    }
    vars(args)['dataset_kwargs'] = {
        'dataset':args.dataset.name,
        'data_dir':args.data_dir,
        'download':args.download,
        'debug_subset_size':args.debug_subset_size if args.debug else None
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last':True,
        'pin_memory':True,
        'num_workers':args.dataset.num_workers
    }

    return args