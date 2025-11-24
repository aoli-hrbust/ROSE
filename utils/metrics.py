# MIT License

# Copyright (c) 2025 Ao Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import logging
import time
from collections import defaultdict
from pathlib import Path as P
from typing import List

import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigs
from sklearn.cluster import k_means
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from torch import Tensor

from .idecutils import cluster_acc
from .idecutils import purity_score,nmi_score
from .kmeans_pytorch import kmeans as kmeans_torch
from .kmeans_pytorch import pairwise_distance as pairwise_distance_torch
from .torch_utils import convert_numpy, convert_tensor
from tqdm import tqdm


_METRICS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    # Rooted MSE
    "rmse": lambda x, y: np.sqrt(mean_squared_error(x, y)),
}

METRICS_LIST = ("ACC", "NMI", "PUR", "F1")




@torch.no_grad()

class MaxMetrics:
    def __init__(self, **kwds):
        """
        MaxMetrics(acc=True, nmi=True, loss=False) 表示一个指标是越大越好，还是越小越好。
        比如 acc、nmi 是越大越好，loss 是越小越好。
        """

        self._data = defaultdict(lambda: -9999)
        self._greater = defaultdict(lambda: True)
        self._current = None
        self._checkpoint = {}
        # 为了保证顺序，必须对kwds 全部进行设置。
        for key, is_greater in kwds.items():
            self._greater[key] = is_greater
            self._data[key] = -9999 if is_greater else 9999

    def update(self, **kwds) -> dict:
        self._current = kwds
        updated = {}
        for key, value in kwds.items():
            gt = self._greater[key]
            if (gt and value > self._data[key]) or (not gt and value < self._data[key]):
                self._data[key] = value
                updated[key] = True
            else:
                updated[key] = False
        return updated

    def report(self, current=False, name=None, compact=False, places=4):
        if current:
            data = self._current
        elif name is not None:
            data = self._checkpoint[name]
        else:
            data = self._data

        data = {k: round(float(v.item() if isinstance(v, torch.Tensor) else v), places) for k, v in data.items()}
        if compact:
            return list(data.values())
        return data

    def save_current_best(self, name):
        self._checkpoint[name] = self._data.copy()



def compute_inertia(X, centroid):
    dist = pairwise_distance_torch(X, centroid)
    min_dist = torch.min(dist, dim=1)[0]
    inertia = torch.sum(min_dist)
    return inertia


def KMeans_Torch(X: Tensor, *, n_clusters, n_init=20, max_iter=1000, verbose=False):
    """
    Return: centroid, ypred
    """
    inertia_best = None
    ypred_best = None
    centroid_best = None
    iter_best = None
    for i in range(n_init):
        ypred, centroid = kmeans_torch(
            X,
            num_clusters=n_clusters,
            device=X.device,
            tqdm_flag=False,
            iter_limit=max_iter,
            # seed=args.seed,
        )
        inertia = compute_inertia(X, centroid)
        if verbose:
            print(f"iter {i:04} inertia {inertia:.4f}")
        if inertia_best is None or inertia < inertia_best:
            iter_best = i
            inertia_best = inertia
            ypred_best = ypred
            centroid_best = centroid

    if verbose:
        print(f"best iter {iter_best:04} inertia {inertia_best:.4f}")
    return centroid_best, ypred_best  # follow sklearn.



def Rose_get_all_metrics(label, ypred):

    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    if isinstance(ypred, torch.Tensor):
        ypred = ypred.cpu().numpy()
    

    if len(label.shape) == 2 and label.shape[1] == 1:
        label = label.squeeze(1)

    if len(ypred.shape) == 2 and ypred.shape[1] == 1:
        ypred = ypred.squeeze(1)
    

    min_len = min(len(label), len(ypred))
    label = label[:min_len] if len(label.shape) == 1 else label[:min_len, :]
    ypred = ypred[:min_len] if len(ypred.shape) == 1 else ypred[:min_len, :]

    metrics = dict(
        ACC=cluster_acc(label, ypred)* 100,
        NMI=nmi_score(label, ypred)* 100,
        PUR=purity_score(label, ypred)* 100,
    )
    return metrics







