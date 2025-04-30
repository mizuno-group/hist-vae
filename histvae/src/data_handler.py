# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

data handler

@author: tadahaya
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import inspect

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# functions
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


class PointHistDataset(Dataset):
    def __init__(
            self, data, group, label=None, transform=False,
            num_points=768, bins=64, noise=None, **transform_params
            ):
        """
        Parameters
        ----------
        data: np.ndarray
            the data to be used for training

        group: np.ndarray
            the group to which the data belongs
        
        label: np.ndarray
            the label of the data
        
        transform: bool
            whether to apply transform to the data

        bins: int
            the number of bins for the histogram

        num_points: int
            the number of points to be sampled

        noise: float
            the noise to be added to the histogram
        
        """
        super().__init__()
        # check the input
        assert data.shape[0] == group.shape[0] == label.shape[0], "!! data, group, and label must have the same number of samples !!"
        self.data = data
        self.group = group
        self.label = label
        self.bins = bins
        self.num_points = num_points
        self.noise = noise or (1 / num_points)
        # tie the group to the data
        self.unique_groups = np.unique(group)
        self.idx2group = {i: j for i, j in enumerate(self.unique_groups)} # map index in the dataset to the group
        self.group2idx = {v: k for k, v in self.idx2group.items()} # map group to index in the dataset
        self.num_data = len(self.unique_groups)
        self.transform = transform
        if transform:
            trans = PCAugmentation(**transform_params)
            self._transform_fxn = trans
        else:
            self._transform_fxn = lambda x, y: (x, y)
        # store normalization parameters
        self.log1p_max = dict()


    def __len__(self):
        return self.num_data


    def __getitem__(self, idx):
        # get the indicated data
        group_idx = self.idx2group[idx]
        selected_indices = np.where(self.group == group_idx)[0]
        pointcloud = self.data[selected_indices]
        # limit the number of points if necessary (random sampling)
        if pointcloud.shape[0] > self.num_points:
            idxs0 = np.random.choice(pointcloud.shape[0], self.num_points, replace=False)
            pointcloud0 = pointcloud[idxs0, :]
            idxs1 = np.random.choice(pointcloud.shape[0], self.num_points, replace=False)
            pointcloud1 = pointcloud[idxs1, :]
        else:
            idxs0 = np.random.choice(pointcloud.shape[0], self.num_points, replace=True)
            pointcloud0 = pointcloud[idxs0, :]
            idxs1 = np.random.choice(pointcloud.shape[0], self.num_points, replace=True)
            pointcloud1 = pointcloud[idxs1, :]
        # prepare histogram
        hist0 = calc_hist(pointcloud0, bins=self.bins)
        hist1 = calc_hist(pointcloud1, bins=self.bins)
        # normalize the histogram
        hist0 = np.log1p(hist0) # log1p for numerical stability
        tmp = np.max(hist0) # store the max value for normalization
        self.log1p_max[idx] = tmp
        hist0 = hist0 / tmp # normalize
        hist1 = np.log1p(hist1) # log1p for numerical stability
        hist1 = hist1 / np.max(hist1) # normalize
        hist0 = torch.tensor(hist0, dtype=torch.float32)
        hist1 = torch.tensor(hist1, dtype=torch.float32)
        # transform
        hist1 = self._transform_fxn(hist1) # hist1 only like translation
        # add channel dimension
        hist0 = hist0.unsqueeze(0)
        hist1 = hist1.unsqueeze(0)
        # prepare label
        if self.label is not None:
            label = self.label[selected_indices][0]
            label = torch.tensor(label, dtype=torch.int64)
        else:
            label = None
        # return the data
        return (hist0, hist1), label
        # hist0, original; hist1, noisy
    

    def transform_on(self, **transform_params):
        """
        transform on

        """
        self.transform = True
        trans = PCAugmentation(**transform_params)
        self._transform_fxn = trans


    def transform_off(self):
        """
        transform off

        """
        self.transform = False
        self._transform_fxn = lambda x, y: (x, y)

# ToDo test this
class PCAugmentation:
    def __init__(self,
                 jitter_sigma=0.01,   # Standard deviation of jitter noise (adjustable)
                 jitter_clip=0.03,    # Maximum jitter noise magnitude
                 scale_range=(0.98, 1.02),  # Scaling range (restricted within ±2%)
                 translate_range=0.05,      # Translation range (small absolute range)
                 ):
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.scale_range = scale_range
        self.translate_range = translate_range

    def jitter(self, points):
        jitter_noise = torch.clamp(
            self.jitter_sigma * torch.randn_like(points),
            -self.jitter_clip,
            self.jitter_clip
        )
        points_jittered = points + jitter_noise
        # Clip negative values to zero since coordinates must be positive
        return torch.clamp(points_jittered, min=0)

    def scale(self, points):
        scale = torch.empty(1).uniform_(*self.scale_range).to(points.device)
        return points * scale

    def translate(self, points):
        translation = torch.empty(points.size(-1)).uniform_(
            -self.translate_range,
            self.translate_range
        ).to(points.device)
        points_translated = points + translation
        # Clip negative values to zero
        return torch.clamp(points_translated, min=0)

    def __call__(self, points):
        points = self.jitter(points)
        points = self.scale(points)
        points = self.translate(points)
        return points


class PointHistDataLoader(DataLoader):
    def __init__(
            self, dataset, batch_size, shuffle=False, num_workers=2,
            pin_memory=True, generator=None, worker_init_fn=None
            ):
        """
        My DataLoader to support the custom dataset
        
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
            worker_init_fn=worker_init_fn,
            )


class DataHandler:
    def __init__(self, config:dict):
        assert isinstance(config, dict), "!! config must be a dictionary !!"
        self.config = config


    def make_dataset(self, data, group, label=None, transform=False):
        """
        make dataset for training and testing

        """
        # check the input
        assert data.shape[0] == group.shape[0], "!! data, group, and label must have the same number of samples !!"
        if label is not None:
            assert data.shape[0] == label.shape[0], "!! data, group, and label must have the same number of samples !!"
        # create dataset
        ds_params = inspect.signature(PointHistDataset.__init__).parameters # diff
        ds_args = {k: self.config[k] for k in ds_params if k in self.config}
        dataset = PointHistDataset(
            data=data,
            group=group,
            label=label,
            transform=transform,
            **ds_args
            )
        return dataset


    def make_dataloader(self, dataset, mode="train"):
        """
        prepare train and test loader
        
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            prepared Dataset instance

        mode: str
            "train" or "test"
                
        """
        # check the input
        assert isinstance(dataset, Dataset), "!! dataset must be a torch.utils.data.Dataset !!"
        assert mode in ["train", "test"], "!! mode must be 'train' or 'test' !!"
        # create dataloader
        dl_params = inspect.signature(PointHistDataLoader.__init__).parameters
        # note: only child class PointHistDataLoader arguments are extracted
        dl_args = {k: self.config[k] for k in dl_params if k in self.config and k not in ["dataset", "shuffle"]}
        shuffle = True if mode == "train" else False
        loader = PointHistDataLoader(dataset=dataset, shuffle=shuffle, **dl_args)
        return loader
    

    def make_lut(self, dataset):
        """
        make lookup table for the dataset

        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            prepared Dataset instance

        Returns
        -------
        lut: dict
            lookup table for the dataset
        
        """
        lut = {}
        for i in range(len(dataset)):
            group = dataset.idx2group[i]
            lut[group] = i
        lut = pd.DataFrame({"group": list(lut.keys()), "idx": list(lut.values())})
        return lut