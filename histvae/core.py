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
from matplotlib import pyplot as plt
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
        # restore original histogram
        hist_list = [dataset[i][0][0].numpy()[0] * dataset.log1p_max[i] for i in indices] # ((hist, hist), label)
        hist_list = [np.exp(h) - 1 for h in hist_list]
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
        self.data = None
        self.group = None
        self.label = None


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
        # register
        self.data = data
        self.group = converted_group
        self.label = converted_label
        return data, converted_group, converted_label


    def get_lut(self) -> pd.DataFrame:
        """
        get lookup table
                
        """
        assert self.lut is not None, "!! fit_transform first !!"
        return self.lut
    

    def check_transform(
            self, raw_data, group, indices:list=[], num_points:int=4096, bins:int=64,
            **plot_params
            ):
        """
        check transform

        Parameters
        ----------
        raw_data: np.ndarray
            the data to be checked

        indices: list
            the list of indices to be checked

        """
        list_hist = []
        list_raw = []
        list_title = []
        for idx in indices:
            # raw data
            mask = np.where(group == idx)[0]
            raw = raw_data[mask]
            # converted data
            hist = self.to_hist(raw_data, group, idx, num_points=num_points, bins=bins)
            # summary
            list_raw.append(raw)
            list_hist.append(hist)
            list_title.append(f"Group_{idx}")
        # plot
        plot_scatter(points_list=list_raw, title_list=list_title, **plot_params)
        plot_hist(hist_list=list_hist, title_list=list_title, **plot_params)


    def to_hist(self, raw_data, group, idx:int, num_points:int=4096, bins=64):
        """
        convert to histogram

        """
        selected_indices = np.where(group == idx)[0]
        pointcloud = raw_data[selected_indices]
        if pointcloud.shape[0] > num_points:
            idxs0 = np.random.choice(pointcloud.shape[0], num_points, replace=False)
            pointcloud0 = pointcloud[idxs0, :]
        else:
            idxs0 = np.random.choice(pointcloud.shape[0], num_points, replace=True)
            pointcloud0 = pointcloud[idxs0, :]
        # prepare histogram
        hist0 = calc_hist(pointcloud0, bins=bins)
        # normalize the histogram
        hist0 = np.log1p(hist0) # log1p for numerical stability
        tmp = np.max(hist0) # store the max value for normalization
        hist0 = hist0 / tmp # normalize
        return hist0


def calc_hist(X, bins=16):
    try:
        s = X.shape[1]
    except IndexError:
        s = 1
    if s == 1:
        hist, _ = np.histogram(X, bins=bins, density=False)
    elif s == 2:
        hist, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=bins, density=False)
    elif s == 3:
        hist, _ = np.histogramdd(X, bins=bins, density=False)
    else:
        raise ValueError("!! Input array must be 1D, 2D, or 3D. !!")
    return hist


def plot_hist(hist_list, output="", **plot_params):
    """
    Plot histograms (1D, 2D).

    Parameters:
    ----------
    hist_list : list of np.ndarray
        List of histograms to plot.

    output : str, optional
        File path to save the plot (default: "", meaning no save).

    **plot_params : dict, optional
        Dictionary containing plot customization options:
            - xlabel (str): Label for x-axis
            - ylabel (str): Label for y-axis
            - title_list (list of str): Titles for each subplot
            - cmap (str): Colormap for 2D histograms
            - aspect (str): Aspect ratio for 2D histograms (default: 'equal')
            - color (str): Bar color for 1D histograms (default: 'royalblue')
            - alpha (float): Transparency for 1D histograms (default: 0.7)
    """
    # Default plot parameters
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
    # merge default and custom params
    params = {**default_params, **plot_params}
    num_plots = len(hist_list)
    nrow, ncol = params["nrow"], params["ncol"]
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    axes = np.atleast_1d(axes).flatten()  # Flatten for easy iteration
    for i, hist in enumerate(hist_list):
        ax = axes[i]
        dim = hist.ndim  # Detect dimensionality
        if dim == 1:
            ax.bar(range(len(hist)), hist, width=0.8, color=params["color"], alpha=params["alpha"])
            ax.set_xlabel(params["xlabel"])
            ax.set_ylabel(params["ylabel"])
            ax.set_title(params["title_list"][i] if params["title_list"] else f'1D Histogram {i+1}')
        elif dim == 2:
            im = ax.imshow(hist.T, origin='lower', cmap=params["cmap"], aspect=params["aspect"])
            fig.colorbar(im, ax=ax, label=params["ylabel"])
            ax.set_xlabel(params["xlabel"])
            ax.set_ylabel(params["ylabel"])
            ax.set_title(params["title_list"][i] if params["title_list"] else f'2D Histogram {i+1}')
        else:
            raise NotImplementedError("Only 1D and 2D histograms are supported.")
    # Remove unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if output:
        plt.savefig(output)
    plt.show()
    plt.close()


def plot_scatter(points_list, output="", **plot_params):
    """
    Plot scatter plots from list of 2D point arrays.

    Parameters
    ----------
    points_list : list of np.ndarray
        List of 2D point arrays (each of shape (N, 2)).

    output : str, optional
        File path to save the plot (default: "", meaning no save).

    **plot_params : dict, optional
        Plot customization options:
            - xlabel (str): Label for x-axis
            - ylabel (str): Label for y-axis
            - title_list (list of str): Titles for each subplot
            - color (str): Point color (default: 'royalblue')
            - alpha (float): Point transparency (default: 0.7)
            - s (float): Point size (default: 10)
            - nrow (int): Number of rows in subplot grid
            - ncol (int): Number of columns in subplot grid
    """
    # Default parameters
    default_params = {
        "nrow": 1,
        "ncol": 3,
        "xlabel": "x",
        "ylabel": "y",
        "title_list": None,
        "color": "royalblue",
        "alpha": 0.7,
        "s": 10
    }
    params = {**default_params, **plot_params}
    num_plots = len(points_list)
    nrow, ncol = params["nrow"], params["ncol"]
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    axes = np.atleast_1d(axes).flatten()  # Flatten to 1D array for iteration
    for i, points in enumerate(points_list):
        ax = axes[i]
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected shape (N, 2), got {points.shape}")
        ax.scatter(points[:, 0], points[:, 1],
                   color=params["color"],
                   alpha=params["alpha"],
                   s=params["s"])
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        ax.set_title(params["title_list"][i] if params["title_list"] else f'Scatter {i+1}')
        ax.grid(True)
    # Remove any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if output:
        plt.savefig(output)
    plt.show()
    plt.close()


import numpy as np
from scipy.spatial import KDTree
from scipy.stats import poisson
from scipy.spatial import ConvexHull
import math

def poisson_denoise(data, percent_radius=0.1, p_threshold=0.05):
    """
    denoising function using poisson distribution
    
    Parameters
    ----------
        data : np.ndarray
            pointcloud data with shape (N, D)

        percent_radius : float or list of float
            radius ratio for each dimension
            When float is given, the same ratio is applied to all dimensions.

        p_threshold : float
            p value for the poisson distribution
            
    Returns
    -------
        mask : np.ndarray (bool)
            True if the point is valid, False if it is noise

    """
    N, D = data.shape
    # convert percent_radius to list
    if isinstance(percent_radius, float):
        percent_radius = [percent_radius] * D
    assert len(percent_radius) == D, "!! percent_radius must be same length as data.shape[1] !!"
    # Calculate the radius for each dimension
    radii = np.ptp(data, axis=0) * np.array(percent_radius)
    # Neighbor search using KDTree (seach point is the center of the ellipsoid)
    # Normalize the data to a unit sphere
    data_scaled = data / radii  # scale each dimension to unit sphere
    tree = KDTree(data_scaled)
    # after scale, the radius is 1.0
    counts = tree.query_ball_point(data_scaled, r=1.0, return_length=True) - 1
    # calculate volume to get expectation lambda
    try:
        hull = ConvexHull(data_scaled)
        volume = hull.volume
    except:
        # When sample size is small, ConvexHull calculation is difficult.
        # Then, approximating volume without ConvexHull
        volume = np.prod(np.ptp(data_scaled, axis=0))
    density = N / volume  # density of the point cloud in the scaled space
    unit_volume = (np.pi ** (D / 2)) / math.gamma(D / 2 + 1)  # volume of unit sphere
    expected_lambda = density * unit_volume  # expected lambda for each point
    # noise detection using poisson distribution
    prob = poisson.cdf(counts, expected_lambda)
    mask = prob >= p_threshold
    return mask