# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

core module
a class specific to the model

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.optim as optim
from schedulefree import RAdamScheduleFree
import numpy as np
import pandas as pd
import os, yaml
import inspect
from datetime import datetime

from .src.models import ConvVAE, LinearHead
from .src.trainer import PreTrainer, FineTuner
from .src.data_handler import PointHistDataset, prep_dataloader, plot_hist
from .src.utils import fix_seed

# Base class
class Core:
    def __init__(
            self, config_path: str=None, train_data=None, test_data=None,
            outdir: str=None, exp_name: str=None, seed: int=42
            ):
        # arguments
        self.config_path = config_path
        self.data = train_data
        self.test_data = test_data
        self.outdir = outdir
        self.exp_name = exp_name
        self.seed = seed
        # initialize
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.trainer = None
        self.optimizer = None
        # loading
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        if exp_name is None:
            exp_name = f"exp-{datetime.today().strftime('%y%m%d')}"
        self.config["exp_name"] = exp_name
        # fix seed
        g, seed_worker = fix_seed(seed, fix_cuda=True)
        self._seed = {"seed": seed, "g": g, "seed_worker": seed_worker}

    def init_model(self):
        raise NotImplementedError
    
    def get_model(self):
        return self.model

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
    
    def get_latent(self):
        raise NotImplementedError

    def check_data(self):
        raise NotImplementedError
    
    def load_model(self, model_path: str, config_path: str=None):
        """ load model """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        # initialize model
        self.init_model()
        # load model
        checkpoint = torch.load(model_path)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])


class HistVAE(Core):
    """ class for training and prediction """
    def __init__(self, mode="pretrain", pretrained_model=None, **kwargs):
        super().__init__(**kwargs)
        # arguments
        self.mode = mode
        assert self.mode in ["pretrain", "finetune"], "!! mode must be pretrain or finetune !!"
        if mode == "finetune" and pretrained_model is None:
            raise ValueError("!! pretrained_model must be given in finetune mode !!")
        self.pretrained_model = pretrained_model
        tmp = [self.config["in_channels"]] + [self.config["bins"]] * (self.config["in_dims"])
        self.config["input_shape"] = tuple(tmp)
        # initialize model
        self.init_model(self.mode)


    def init_model(self, mode="pretrain"):
        """
        prepare model
        hard coded parameters

        Parameters
        ----------
        mode: str
            "pretrain" or "finetune"
            note that the model is initialized as ConvVAE in pretraining
            and as LinearHead in finetuning
        
        """
        if mode == "pretrain":
            # prepare pretraining model
            model_params = inspect.signature(ConvVAE.__init__).parameters # diff
            model_args = {k: self.config[k] for k in model_params if k in self.config}
            self.model = ConvVAE(**model_args)
            for param in self.model.parameters():
                param.requires_grad = True
            optimizer = RAdamScheduleFree(self.model.parameters(), lr=float(self.config["lr"]), betas=(0.9, 0.999))
            self.trainer = PreTrainer(
                self.config, self.model, optimizer, outdir=self.outdir
                )
            self.optimizer = optimizer
        elif mode == "finetune":
            # prepare linear head
            model_params = inspect.signature(LinearHead.__init__).parameters # diff
            model_args = {k: self.config[k] for k in model_params if k in self.config}
            self.model = LinearHead(self.pretrained_model, **model_args)
            for param in self.model.parameters():
                param.requires_grad = True
            optimizer = RAdamScheduleFree(self.model.parameters(), lr=float(self.config["lr"]), betas=(0.9, 0.999))
            loss_fn = nn.CrossEntropyLoss() # hard coded
            self.trainer = FineTuner(
                self.config, self.model, optimizer, loss_fn, outdir=self.outdir
                )
            self.optimizer = optimizer


    def prep_data(
            self, train_data=None, train_group=None, train_label=None,
            test_data=None, test_group=None, test_label=None
            ):
        """ prepare data """
        # force no transform when finetuning
        self.train_dataset = PointHistDataset(
            data=train_data,
            group=train_group,
            label=train_label,
            mode=self.mode,
            num_points=self.config["num_points"],
            bins=self.config["bins"],
        )
        train_loader = prep_dataloader(
            self.train_dataset, self.config["batch_size"], True, self.config["num_workers"],
            self.config["pin_memory"], self._seed["g"], self._seed["seed_worker"]
            )
        if test_data is None:
            return train_loader, None
        else:
            self.test_dataset = PointHistDataset(
                data=test_data,
                group=test_group,
                label=test_label,
                mode=self.mode,
                num_points=self.config["num_points"],
                bins=self.config["bins"],
            )
            test_loader = prep_dataloader(
                self.test_dataset, self.config["batch_size"], False, self.config["num_workers"],
                self.config["pin_memory"], self._seed["g"], self._seed["seed_worker"]
                )
            return train_loader, test_loader


    def train(self, train_loader, test_loader, callbacks:list=None, verbose:bool=True):
        """ training """
        if callbacks is not None:
            self.trainer.set_callbacks(callbacks)
        self.trainer.train(train_loader, test_loader)
        if verbose:
            print(">> Training is done.")


    # ToDo: check this
    def predict(self, data_loader=None):
        """ prediction """
        if data_loader is None:
            raise ValueError("!! Give data_loader !!")
        if self.model is None:
            raise ValueError("!! fit or load_model first !!")
        self.finetuned_model.eval()
        preds = []
        probs = []
        labels = []
        with torch.no_grad():
            for data, label in data_loader:
                hist0, hist1 = (x.to(self.device) for x in data)
                label = label.to(self.device)
                logits, recon, mu, logvar = self.finetuned_model(hist0) # use original hist
                preds.append(logits.argmax(dim=1).cpu().numpy())
                probs.append(logits.cpu().numpy())
                labels.append(label.cpu().numpy())
        return np.concatenate(preds), np.concatenate(probs), np.concatenate(labels)


    def get_latent(self, dataset=None, indices:list=[]):
        """
        get latent representation
        note: pretrained model weight is changed after finetuning.
        
        """
        if dataset is None:
            dataset = self.test_dataset
        if self.model is None:
            raise ValueError("!! fit or load_model first !!")
        self.model.eval()
        num_data = len(dataset)
        if len(indices) == 0:
            indices = list(range(num_data))
        reps = []
        with torch.no_grad():
            for i in indices:
                data, _ = dataset[i]
                hist0, hist1 = (x.to(self.device).unsqueeze(0) for x in data)  # add batch dimension
                mu, logvar = self.model.encode(hist0) # use original hist
                # note both ConvVAE and LinearHead have encode method
                reps.append(mu.cpu().numpy().reshape(1, -1))  # del batch dimension
        return np.vstack(reps)


    def check_data(self, dataset, indices:list=[], output:str="", **plot_params):
        """
        check data
        
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            the PHTwins dataset

        indices: list
            the list of indices to be checked

        output: str
            the output path

        plot_params: dict
            the parameters for the plot
            default_params = {
                "nrow": 1,
                "ncol": 3,
                "xlabel": "x",
                "ylabel": "y",
                "title_list": None,
                "cmap": "viridis",
                "aspect": "equal",
                "color": "royalblue",
                "alpha": 0.7
            }
        
        """
        hist_list = [dataset[i][0][0].numpy()[0] for i in indices] # ((hist, hist), label)
        plot_hist(hist_list, output, **plot_params)


    def qual_eval(self, dataset, query_indices, outdir:str=""):
        """
        qualitative evaluation
        
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            the PHTwins dataset

        indices: list
            the list of indices to be checked
        
        """
        # get representations
        reps = self.get_latent(dataset) # default: train dataset
        # query data
        query_reps = reps[query_indices]
        # calculate cosine similarity
        norm_query = np.linalg.norm(query_reps, axis=1, keepdims=True)
        norm_reps = np.linalg.norm(reps, axis=1)
        norm_query[norm_query == 0] = 1e-10
        norm_reps[norm_reps == 0] = 1e-10
        sim_matrix = np.dot(query_reps, reps.T) / (norm_query * norm_reps)
        # plot query, most similar, and least similar
        for i, idx in enumerate(query_indices):
            output = os.path.join(outdir, f"qual_eval_{i}.tif")
            indices = np.argsort(sim_matrix[i])[::-1]
            plot_indices = [idx] + [indices[0]] + [indices[-1]]
            plot_params = {
                "title_list": ["query", "most similar", "least similar"],
                "nrow": 1,
                "ncol": 3,
                }
            self.check_data(dataset, plot_indices, output, **plot_params)
            # nrows, ncols = 1, 3 (query / most similar / least similar)
        return sim_matrix

class Preprocess:
    def __init__(
            self, key_data, key_group, key_label=None
            ):
        """
        Parameters
        ----------
        key_data: list
            the keys for the data

        key_group: str
            the key to identify the data

        key_label: int
            the key for the label
            note that the label should be integer
        
        """
        self.key_data = key_data
        self.key_group = key_group
        self.key_label = key_label
        self.idx2id = None
        self.id2idx = None
        self.idx2label = None
        self.label2idx = None
        self.num_data = None


    def fit_transform(self, df):
        """
        preprocess the data

        Parameters
        ----------
        df: pd.DataFrame
            the data to be preprocessed

        Returns
        -------
        data: np.ndarray
            preprocessed data
        
        group: np.ndarray
            preprocessed group

        label: np.ndarray
            preprocessed label

        """
        # prepare meta data and converter
        # group, assumed to be 1D
        group = df[self.key_group].values.flatten()
        unique_group = np.unique(group)
        self.idx2id = {k: v for k, v in enumerate(unique_group)} # index to identifier
        self.id2idx = {v: k for k, v in self.idx2id.items()} # identifier to index
        self.num_data = len(unique_group) # number of data
        converted_group = np.array([self.id2idx[k] for k in group])
        # label, assumed to be 1D
        if self.key_label is not None:
            label = df[self.key_label].values.flatten()
            unique_label = np.unique(label)
            self.idx2label = {k: v for k, v in enumerate(unique_label)} # index to label
            self.label2idx = {v: k for k, v in self.idx2label.items()} # label to index
            converted_label = np.array([self.label2idx[k] for k in label])
        else:
            self.idx2label = None
            self.label2idx = None
            converted_label = None
        # data
        data = df[self.key_data].values
        data = data.astype(np.float32)
        return data, converted_group, converted_label


    def get_meta(self) -> pd.DataFrame:
        """
        get meta data that contains:
            - group indices (used in training)
            - group values
            - label indices (used in training)
            - label values
        
        """
        meta = pd.DataFrame({
            "group_indices": list(self.idx2id.keys()),
            "group_values": list(self.idx2id.values()),
            "label_indices": list(self.idx2label.keys()),
            "label_indices": list(self.idx2id.values()),
            })
        return meta