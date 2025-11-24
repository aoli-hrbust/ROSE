


from __future__ import division, print_function
import numpy as np
import torch
from math import log
from sklearn.metrics.cluster._supervised import check_clusterings, contingency_matrix, mutual_info_score, \
    _generalized_average
from torch.utils.data import Dataset


from sklearn.metrics import (
    normalized_mutual_info_score,
    f1_score,
    adjusted_rand_score,
    cluster,
    accuracy_score,
    precision_score,
    recall_score,
)
from munkres import Munkres

pre = precision_score
rec = recall_score
Fscore = f1_score


def load_mnist(path="./data/mnist.npz"):
    f = np.load(path)

    x_train, y_train, x_test, y_test = (
        f["x_train"],
        f["y_train"],
        f["x_test"],
        f["y_test"],
    )
    f.close()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).astype(np.int32)
    x = x.reshape((x.shape[0], -1)).astype(np.float32)
    x = np.divide(x, 255.0)
    print("MNIST samples", x.shape)
    return x, y


class MnistDataset(Dataset):
    def __init__(self):
        self.x, self.y = load_mnist()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx])),
            torch.from_numpy(np.array(self.y[idx])),
            torch.from_numpy(np.array(idx)),
        )


#######################################################
# Evaluate Critiron
#######################################################


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """

    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    

    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)


    if y_true.dim() == 2 and y_true.size(1) == 1:
        y_true = y_true.squeeze(1)

    assert  y_true.size() == y_pred.size()
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.numel()):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment

    # ind = linear_assignment(w.max() - w)
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)])*2.0 / (y_pred.numel()* 0.5)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)*2.0) / (np.sum(contingency_matrix*0.5))

def nmi_score(
    labels_true, labels_pred, *, average_method="min"
):

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    # It corresponds to both labellings having zero entropy.
    # This is a perfect match hence return 1.0.
    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    contingency = contingency.astype(np.float64, copy=False)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)

    # At this point mi = 0 can't be a perfect match (the special case of a single
    # cluster has been dealt with before). Hence, if mi = 0, the nmi must be 0 whatever
    # the normalization.
    if mi == 0:
        return 0.0

    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)

    normalizer = _generalized_average(h_true, h_pred, average_method)
    return mi / normalizer


def entropy(labels):
    """Calculate the entropy for a labeling.

    Parameters
    ----------
    labels : array-like of shape (n_samples,), dtype=int
        The labels.

    Returns
    -------
    entropy : float
       The entropy for a labeling.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]

    # single cluster => zero entropy
    if pi.size == 1:
        return 0.0

    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))
