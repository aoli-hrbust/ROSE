import itertools
import json
import logging
import pickle
import time
from pathlib import Path as P
from pprint import pformat
from typing import Tuple, Union
import jsons
import matplotlib.pyplot as plt
from typing import Literal, List

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import subprocess
import os
import sys


def encode_path(**kwargs):
    items = sorted(kwargs.items(), key=lambda x: x[0])
    return "-".join([f"{k}={v}" for k, v in items])

def kv_product(**kwargs):
    """
    >>> for kwargs in kv_product(a='abc', b='xyz'): print(kwargs)
    """
    for val in itertools.product(*kwargs.values()):
        yield dict(zip(kwargs.keys(), val))


def save_var(savedir: P, var, name: str):
    """
    Save a single variable to savedir.
    """
    savedir.mkdir(exist_ok=1, parents=1)
    f = savedir.joinpath(name).with_suffix(".pkl")
    pickle.dump(var, f.open("wb"))
    logging.info(f"Save Var to {f}")


def save_variables(savedir: P, variables: dict):
    for key, val in variables.items():
        save_var(savedir, val, key)

def save_json(savedir: P, var, name: str):

    savedir = P(savedir)
    savedir.mkdir(exist_ok=1, parents=1)
    f = savedir.joinpath(name).with_suffix(".json")
    f.write_text(jsons.dumps(var, jdkwargs=dict(indent=4)))
    logging.info(f"Save Var to {f}")


def train_begin(savedir: P, config: dict, message: str = None):
    message = message or "Train begins\n"
    logging.info(f"{message} {pformat(config)}")
    save_json(savedir, config, "config")


def train_end(savedir: P, metrics: dict, message: str = None):
    message = message or "Train ends"
    logging.info(f"{message} {metrics}")
    save_json(savedir, metrics, "metrics")

