# MIT License

# Copyright (c) 2023 Ao Li

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

import itertools
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from pandas import DataFrame as DF
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances

from datasets.dataset import PartialMultiviewDataset



def get_images(X, size: int = 32):
    return X.reshape([-1, 32, 32]).transpose([0, 2, 1])


def show_gnd_hat(num: int, x_gnd, x_hat, mask):
    row = 2
    col = num
    scale = 2

    image_gnd = get_images(x_gnd)[mask]
    image_hat = get_images(x_hat)[mask]

    sampleNum = len(image_gnd)
    samples = np.arange(sampleNum, dtype=int)
    np.random.shuffle(samples)
    samples = samples[:num]

    plt.figure(figsize=(col * scale, row * scale))
    for i, hat, gnd in zip(range(num), image_hat[samples], image_gnd[samples]):
        plt.subplot(2, num, i + 1)
        plt.imshow(hat)
        # plt.title(f"hat {i}")
        plt.subplot(2, num, num + i + 1)
        plt.imshow(gnd)
        # plt.title(f"gnd {i}")


def plot_gnd_hat(X_gnd, X_hat, pdist=False, cmap='gray'):
    """Display sidebyside x_gnd and x_hat"""
    assert len(X_gnd) == len(X_hat)
    rows = 2
    cols = len(X_gnd)
    scale = 5

    if pdist:  # Plot distance matrix instead.
        dis = Parallel(n_jobs=-1, verbose=1)(
            delayed(pairwise_distances)(x) for x in itertools.chain(X_gnd, X_hat)
        )
        X_gnd, X_hat = dis[:cols], dis[cols:]

    plt.figure(figsize=((cols + 1) * scale, rows * scale))
    for v, (x_gnd, x_hat) in enumerate(zip(X_gnd, X_hat), start=1):
        sns.heatmap(x_gnd, ax=plt.subplot(rows, cols, v), cbar=False, xticklabels=False, yticklabels=False, cmap=cmap)
        # plt.title(f"x_gnd {v}")

        sns.heatmap(x_hat, ax=plt.subplot(rows, cols, v + cols), cbar=False, xticklabels=False, yticklabels=False, cmap=cmap)
        # plt.title(f"x_hat {v}")
    plt.tight_layout()


