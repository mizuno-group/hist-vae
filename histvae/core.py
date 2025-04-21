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

from .src.models import ModelHandler
from .src.trainer import PreTrainer, FineTuner
from .src.data_handler import DataHandler, plot_hist
from .src.utils import fix_seed

class HistVAE:
    def __init__(
            self, config: dict=None, outdir: str=None, exp_name: str=None, seed: int=42
            ):
        # arguments
        assert config is not None, "!! config must be given !!"
        self.config = config
        self.outdir = outdir
        self.exp_name = exp_name
        self.seed = seed
        # delegate
        self.data_handler = DataHandler(config)
        self.model_handler = ModelHandler(config)
        # initialize
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.train_lut = None
        self.test_lut = None
        self.model = None
        self.trainer = None
        self.optimizer = None
        self.loss_fn = None
        # fix seed
        g, seed_worker = fix_seed(seed, fix_cuda=True)
        self._seed = {"seed": seed, "generator": g, "worker_init_fn": seed_worker}
        # loading
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if exp_name is None:
            exp_name = f"exp-{datetime.today().strftime('%y%m%d')}"
        self.config["exp_name"] = exp_name
        tmp = [self.config["in_channels"]] + [self.config["bins"]] * (self.config["in_dims"])
        self.config["input_shape"] = tuple(tmp) # hard coded for ConvVAE


    def prep_data(
            self, train_data=None, train_group=None, train_label=None, train_transform=None,
            test_data=None, test_group=None, test_label=None, test_transform=None
            ):
        """ prepare data """
        if train_transform is None:
            train_transform = self.config.get("transform_2d", True)
        if test_transform is None:
            test_transform = False
        # dataset
        self.train_dataset = self.data_handler.make_dataset(
            data=train_data, group=train_group, label=train_label, transform=train_transform
            )
        if test_data is not None:
            self.test_dataset = self.data_handler.make_dataset(
                data=test_data, group=test_group, label=test_label, transform=test_transform
                )
        # dataloader
        self.train_loader = self.data_handler.make_dataloader(dataset=self.train_dataset, mode="train")
        if self.test_dataset is not None:
            self.test_loader = self.data_handler.make_dataloader(dataset=self.test_dataset, mode="test")
        # lookup table
        self.train_lut = self.data_handler.make_lut(dataset=self.train_dataset)
        self.test_lut = None
        if self.test_dataset is not None:
            self.test_lut = self.data_handler.make_lut(dataset=self.test_dataset)


    def prep_model(self, mode="pretrain", model_path:str=None):
        """
        prepare model
        hard coded parameters

        Parameters
        ----------
        mode: str
            "pretrain", "cpt", or "finetune"

        model_path: str
            path to the pretrained model
            only used in "cpt" and "finetune"
        
        """
        # check the mode
        assert mode in ["pretrain", "cpt", "finetune"], "!! mode must be pretrain, cpt, or finetune !!"
        if mode == "pretrain":
            # prepare pretraining model
            self.model = self.model_handler.make_pretrain()
            self.optimizer = RAdamScheduleFree(
                self.model.parameters(), lr=float(self.config["lr"]), betas=(0.9, 0.999),
                weight_decay=float(self.config["weight_decay"])
                )
            self.trainer = PreTrainer(
                self.config, self.model, self.optimizer, outdir=self.outdir
                )
        elif mode == "cpt":
            # prepare continuous pretraining model
            assert model_path is not None, "!! model_path must be given in cpt mode!!"
            self.model = self.model_handler.make_cpt(model_path=model_path)
            self.optimizer = RAdamScheduleFree(
                self.model.parameters(), lr=float(self.config["lr"]), betas=(0.9, 0.999),
                weight_decay=float(self.config["weight_decay"])
                )
            self.trainer = PreTrainer(
                self.config, self.model, self.optimizer, outdir=self.outdir
                )
        elif mode == "finetune":
            # prepare finetuning model
            assert model_path is not None, "!! model_path must be given in finetune mode!!"
            self.model = self.model_handler.make_finetune(model_path=model_path)
            self.optimizer = RAdamScheduleFree(
                self.model.parameters(), lr=float(self.config["lr"]), betas=(0.9, 0.999),
                weight_decay=float(self.config["weight_decay"])
                )
            self.loss_fn = nn.CrossEntropyLoss()
            self.trainer = FineTuner(
                self.config, self.model, self.optimizer, self.loss_fn, outdir=self.outdir
                )


    def train(self, callbacks:list=None, verbose:bool=True):
        """ training """
        if callbacks is not None:
            self.trainer.set_callbacks(callbacks)
        self.trainer.train(self.train_loader, self.test_loader)
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


    # ToDo: check this
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
        dataset.transform_off() # disable transform
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


    # ToDo: check this
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


    # ToDo: check this
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
        self.lut = None
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
        # group and label are assumed to be 1D
        if self.key_label is None:
            group = df[self.key_group].values
            unique_group = np.unique(group)
            self.lut = pd.DataFrame(
                {"raw_index":list(range(len(unique_group))), "group": unique_group}
                )
            self.lut[:, "label"] = None
            converted_label = None
        else:
            tmp = dict(zip(list(df[self.key_group]), list(df[self.key_label])))
            self.lut = pd.DataFrame(
                {"raw_index":list(range(len(tmp))), "group": list(tmp.keys()), "label": list(tmp.values())}
                )
            dic_label = dict(zip(list(self.lut["label"]), list(self.lut["raw_index"])))
            converted_label = np.array([dic_label[k] for k in list(df[self.key_label])])
        dic_group = dict(zip(list(self.lut["group"]), list(self.lut["raw_index"])))
        converted_group = np.array([dic_group[k] for k in list(df[self.key_group])])
        self.num_data = self.lut.shape[0] # number of data
        # data
        data = df[self.key_data].values
        data = data.astype(np.float32)
        return data, converted_group, converted_label


    def get_lut(self) -> pd.DataFrame:
        """
        get lookup table
                
        """
        assert self.lut is not None, "!! fit_transform first !!"
        return self.lut