# MIT License
import logging
import traceback

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

from collections import Counter
from os import PathLike
from pathlib import Path as P, Path
from typing import Any, Dict, List, Union

import h5py
import numpy as np
import scipy.io as sio
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, normalize, scale

from .get_mask import get_mask


def try_get_item(data: Dict[str, Any], key_list: List[str]):
    for key in key_list:
        if key in data:
            return data[key]
    raise KeyError(f"No such key: {data.keys()}")


NORMALIZE_CLASS = {
    "minmax": pp.MinMaxScaler,
    "center": pp.StandardScaler,
    "l2": pp.Normalizer,
}


def num_per_class(Y) -> List[int]:
    return list(Counter(Y).values())


class MultiviewDataset:
    """Basic logic of loading a multiview dataset from .mat file."""

    def __init__(
        self,
        datapath: P,  # full path to mat file.
        normalize: str = None,  # normalization strategy.
        view_ids=None,  # subset of views to use.
        shuffle=False,
    ):
        self.describe_excludes = [
            "X",
            "Y",
            "describe_excludes",
            "cluster_sizes",
            "scalers",
            # "feature_names",
            # "class_names",
        ]
        datapath = P(datapath)
        # Load raw data from .mat file.
        try:
            data = sio.loadmat(str(datapath), simplify_cells=True)
        except NotImplementedError:
            data = h5py.File(str(datapath))
        # print(f'Raw mat keys: {data.keys()}')

        # Extract X and Y.
        if "X_names" in data:
            X_names = data["X_names"]
            X = [data[str(name)] for name in X_names]
        else:
            X = list(try_get_item(data, ["X", "data"]))

        if "Y_name" in data:
            Y_name = data["Y_name"]
            Y = data[Y_name]
        else:
            Y = try_get_item(data, ["truth", "Y", "gt", "gnd", "truelabel", "y"])

        try:
            self.feature_names = try_get_item(data, "view_meaning feanames".split())
        except KeyError:
            self.feature_names = None

        try:
            self.class_names = try_get_item(data, "class_meaning ".split())
        except KeyError:
            self.class_names = None

        if Y.dtype == object:
            Y = Y[0]

        # make sure y is zero-based (from matlab often is one-based.)
        if np.min(Y) == 1:
            Y -= 1

        # Type conversion.
        if type(X[0]) != np.ndarray:
            # This may be a sparse matrix.
            X = [x.astype(np.float32).toarray() for x in X]
        else:
            X = [x.astype(np.float32) for x in X]

        # Fix problem of transpose.
        sampleNum = len(Y)
        column_instance = all(x.shape[1] == sampleNum for x in X)
        if column_instance:
            # Make sure data are row-instance.
            X = [x.T for x in X]

        # Perform view selection.
        if view_ids is not None:
            # Select a subset of views.
            X = [X[v] for v in view_ids]

        # Normalize data.
        if normalize is not None:
            scalers = [NORMALIZE_CLASS[normalize]() for _ in X]
            X = [scaler.fit_transform(x) for scaler, x in zip(scalers, X)]
        else:
            scalers = None

        if shuffle:
            logging.info('Dataset: shuffle=True')
            import random

            rand_idx = list(range(sampleNum))
            random.shuffle(rand_idx)
            for v in range(len(X)):
                X[v] = X[v][rand_idx]
            Y = Y[rand_idx]
            
        self.X = X
        self.Y = Y
        self.sampleNum = sampleNum
        self.viewNum = len(X)
        self.name = datapath.stem
        self.view_ids = view_ids
        self.view_dims = [x.shape[1] for x in X]
        self.cluster_sizes = num_per_class(Y)
        self.clusterNum = len(np.unique(Y))
        self.normalize = normalize
        self.scalers = scalers
        self.datapath = P(datapath)

    def describe(self):
        return dict(
            filter(
                lambda item: not any(
                    item[0].startswith(s) for s in self.describe_excludes
                ),
                self.__dict__.items(),
            )
        )

    def short_name(self):
        return f"{self.name}-{self.normalize}"

    def save(self, outfile: P):
        X = np.array([None] * self.viewNum, dtype=object)
        for v in range(self.viewNum):
            X[v] = self.X[v]
        savemat(str(outfile), X, self.Y)


class PartialMultiviewDataset(MultiviewDataset):
    def __init__(
        self,
        datapath: PathLike,
        paired_rate: float,
        normalize: str = None,
        view_ids=None,
        partial_kind=None,  # general, partial (default)
        shuffle=False,
    ):
        super().__init__(datapath=datapath, normalize=None, view_ids=view_ids, shuffle=shuffle)
        if "MSRCv1" in str(datapath) and paired_rate == 0.9:
            paired_rate = 0.85

        self.describe_excludes.extend(["mask", "idx"])
        if partial_kind is None:
            partial_kind = "partial"

        mask = make_mask(
            paired_rate=paired_rate,
            sampleNum=self.sampleNum,
            viewNum=self.viewNum,
            kind=partial_kind,
        )

        assert mask.dtype == bool

        mask_sum: np.ndarray = np.sum(mask, 1)
        # Aligned samples
        mask_paired = mask_sum == self.viewNum
        # Existent samples (including aligned ones)
        mask_all_list = [msk for msk in mask.T]
        assert all(
            np.count_nonzero(m_all) >= np.count_nonzero(mask_paired)
            for m_all in mask_all_list
        )
        # Unaligned samples (but existent)
        mask_single_list = [
            mask_all_list[v] & np.logical_not(mask_paired) for v in range(self.viewNum)
        ]


        idx_paired = np.nonzero(mask_paired)[0]
        idx_single_list = [np.nonzero(m_single)[0] for m_single in mask_single_list]
        idx_all_list = [np.r_[idx_paired, idx] for idx in idx_single_list]

        # The original complete data.
        X_gnd = self.X
        # Data with missing views (set to zeros).
        X = [
            np.where(np.expand_dims(msk, 1), x, np.nan) for msk, x in zip(mask.T, X_gnd)
        ]
        from sklearn.impute import SimpleImputer

        X = [SimpleImputer(strategy="mean").fit_transform(x) for x in X]
        if normalize is not None:
            assert self.scalers is None
            self.normalize = normalize
            self.scalers = self.init_scalers()
            for x, idx, scaler in zip(X, idx_all_list, self.scalers):
                x[idx] = scaler.fit_transform(x[idx])
            X_gnd = [scaler.transform(x) for scaler, x in zip(self.scalers, X_gnd)]

        X_all = [x[idx] for idx, x in zip(idx_all_list, X)]
        X_paired = [x[idx_paired] for x in X]
        X_single = [x[idx] for idx, x in zip(idx_single_list, X)]

        self.partial_kind = partial_kind
        self.paired_rate = paired_rate
        self.pairedNum = len(idx_paired)
        self.mask = mask
        self.mask_not = np.logical_not(mask)
        self.mask_sum = mask_sum
        self.X = X
        self.X_list = X  # Alias to X
        self.X_gnd = X_gnd
        self.X_all_list = X_all
        self.X_paired_list = X_paired
        self.X_single_list = X_single
        self.idx_all_list = idx_all_list
        self.idx_single_list = idx_single_list

    def init_scalers(self):
        return [NORMALIZE_CLASS[self.normalize]() for _ in range(self.viewNum)]

    def inverse_scale(self, X_scaled):

        if self.scalers is None:
            print("没有进行normalize，不需要inverse_scale")
            return X_scaled
        return [
            scaler.inverse_transform(x) for scaler, x in zip(self.scalers, X_scaled)
        ]

    def short_name(self):
        return f"{self.name}-{self.paired_rate * 100}-{self.normalize}"


def mask_general_multiview(*, paired_rate: float, sampleNum: int, viewNum: int):

    return get_mask(
        view_num=viewNum, data_len=sampleNum, missing_rate=(1 - paired_rate) * viewNum
    ).astype(bool)


def mask_weaker_multiview(*, paired_rate: float, sampleNum: int, viewNum: int):

    return get_mask(
        view_num=viewNum, data_len=sampleNum, missing_rate=(1 - paired_rate)
    ).astype(bool)


def mask_partial_multiview(*, paired_rate, sampleNum, viewNum):
    pairedSmp, partialSmp = train_test_split(
        np.arange(sampleNum),
        train_size=paired_rate,
        shuffle=True,
    )
    partial_mask = get_mask(
        view_num=viewNum,
        data_len=len(partialSmp),
        missing_rate=1,
    )
    ind = np.zeros([sampleNum, viewNum])
    ind[pairedSmp, :] = 1
    ind[partialSmp, :] = partial_mask
    return ind.astype(bool)


_MASK_FNS = dict(
    general=mask_general_multiview,
    partial=mask_partial_multiview,
    weaker=mask_weaker_multiview,
)


def make_mask(*, paired_rate: float, sampleNum: int, viewNum: int, kind: str):


    if paired_rate == 1.0:

        return np.ones([sampleNum, viewNum], dtype=bool)
    try:
        return _MASK_FNS[kind](
            paired_rate=paired_rate, sampleNum=sampleNum, viewNum=viewNum
        )
    except Exception as e:
        logging.warning(f"Exception with kind `{kind}`. Fall back to general kind")
        # logging.warning(traceback.format_exception(e))
        kind = "general"
        return _MASK_FNS[kind](
            paired_rate=paired_rate, sampleNum=sampleNum, viewNum=viewNum
        )

def savemat(filename: str, X: List[np.ndarray], Y: np.ndarray):
    # This is the robust way to save a cell.
    sio.savemat(str(filename), dict(X=X, Y=Y), do_compression=True)


if __name__ == "__main__":
    make_mask(
        paired_rate=0.5,
        kind="general",
        sampleNum=100,
        viewNum=4,
    )
