# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

main file

@author: tadahaya
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from .data_handler import prep_data
from .models import VitForClassification
from .trainer import Trainer
from .utils import load_config


def get_args():
    """ 引数の取得 """
    parser = argparse.ArgumentParser(description="Yaml file for training")
    parser.add_argument("--config_path", type=str, default=None, help="Yaml file for training (defaults to packaged config.yaml)")
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
    config, config_meta = load_config(
        config_path=args.config_path,
        overrides={
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "exp_name": args.exp_name,
        },
    )
    config["config_path"] = config_meta["user_config_path"] or config_meta["default_config_path"]
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
