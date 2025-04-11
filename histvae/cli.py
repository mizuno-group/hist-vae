# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

main file

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import argparse
import yaml

from tqdm.auto import tqdm

from .src.arguments import get_args
from .src.models import *
from .src.trainer import Trainer
from .src.data_handler import prep_data, prep_test


def get_args():
    """ 引数の取得 """
    parser = argparse.ArgumentParser(description="Yaml file for training")
    parser.add_argument("--config_path", type=str, required=True, help="Yaml file for training")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--input_path", type=str, default=None, help="input data path")
    parser.add_argument("--input_path2", type=str, default=None, help="input data path, test data")
    args = parser.parse_args()
    return args


def test():
    """ test """
    raise NotImplementedError("!! Not implemented yet !!")


def main():
    # argsの取得
    args = get_args()
    # input_pathのチェック
    if args.input_path is None:
        raise ValueError("!! Give input_path !!")
    # yamlの読み込み
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["config_path"] = args.config_path
    config["exp_name"] = args.exp_name
    # dataの読み込み
    train_loader, test_loader, classes = prep_data(
        image_path=(args.input_path, args.input_path2), 
        batch_size=config["batch_size"], transform=(None, None), shuffle=(True, False)
        )
    # モデル等の準備
    model = VitForClassification(config)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2) # AdamW使っている
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(config, model, optimizer, loss_fn, args.exp_name, device=config["device"])
    trainer.train(
        train_loader, test_loader, classes, save_model_evry_n_epochs=config["save_model_every"]
        )
    if args.input_path2 is None:
        accuracy, avg_loss = trainer.evaluate(test_loader)
        print(f"Accuracy: {accuracy} // Average Loss: {avg_loss}")


if __name__ == "__main__":
    main()