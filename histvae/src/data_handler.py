# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

data handler

@author: tadahaya
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

def calc_hist(X, bins=16) -> np.ndarray:
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
            self, df, key_identify, key_data, key_label, num_points=768, bins=16,
            noise=None, transform=None
            ):
        """
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the data and label

        key_identify: str
            the key to identify the data

        key_data: list
            the keys for the data

        key_label: int
            the key for the label
            note that the label should be integer

        bins: int
            the number of bins for the histogram

        num_points: int
            the number of points to be sampled

        noise: float
            the noise to be added to the histogram

        transform: callable
            the transform function to be applied to the data
        
        """
        self.df = df
        self.bins = bins
        self.num_points = num_points
        self.noise = noise
        if noise is None:
            self.noise = 1 / num_points
        self.transform = transform
        # prepare meta data
        self.key_identify = key_identify
        identifier = list(df[key_identify].unique())
        self.idx2id = {k: v for k, v in enumerate(identifier)} # index to identifier
        self.num_data = len(identifier) # number of data
        # prepare data
        self.key_data = key_data
        # prepare label
        self.key_label = key_label
        self.id2label = dict(zip(df[key_identify], df[key_label]))

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, idx) -> tuple:
        # get data
        key = self.idx2id[idx]
        data = self.df[self.df[self.key_identify] == key]
        # get point cloud
        pointcloud = np.array(data[self.key_data], dtype=np.float32)
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
        # transform
        if self.transform:
            pointcloud0 = self.transform(pointcloud0)
            pointcloud1 = self.transform(pointcloud1)
        # prepare histogram
        hist0 = calc_hist(pointcloud0, bins=self.bins) / self.num_points
        hist1 = calc_hist(pointcloud1, bins=self.bins) / self.num_points
        hist1 = self.add_noise(hist1, self.noise) # add noise to the histogram
        hist0 = torch.tensor(hist0, dtype=torch.float32).unsqueeze(0) # add channel dimension
        hist1 = torch.tensor(hist1, dtype=torch.float32).unsqueeze(0) # add channel dimension
        # prepare label
        label = self.id2label[key]
        try:
            label = torch.tensor(label, dtype=torch.long)
        except ValueError:
            pass # if label is None
        return (hist0, hist1), label
    
    def get_meta(self) -> pd.DataFrame:
        """
        get meta data that contains:
            - index
            - identifier
            - label
        
        """
        meta = pd.DataFrame({
            "index": list(range(self.num_data)),
            "identifier": list(self.idx2id.values()),
            "label": [self.id2label[k] for k in self.idx2id.values()]
            })
        return meta


    def add_noise(self, hist, noise=0.001):
        """
        add noise to the histogram
        """
        noise = np.random.normal(0, noise, hist.shape)
        noise = np.where(hist > 0, noise, 0.0)
        hist += noise
        return np.clip(hist, 0.0, None)


def prep_dataloader(
    dataset, batch_size, shuffle=None, num_workers=2, pin_memory=True,
    g=None, seed_worker=None
    ) -> DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        worker_init_fn=seed_worker,
        )    
    return loader