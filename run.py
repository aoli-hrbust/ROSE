import traceback

from models.Rose import train_main
from pathlib import Path as P
import logging
import numpy as np
import warnings
import itertools
from utils.io_utils import *
from joblib import Parallel, delayed
from tqdm import tqdm
from utils.torch_utils import get_device
import warnings



warnings.filterwarnings("ignore")

warnings.filterwarnings(action="ignore")
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    root = P("./data/")


    def func(kwargs: dict):
        savedir = root.joinpath(encode_path(**kwargs))
        if savedir.exists() and savedir.joinpath("metrics.json").exists():
            return

        dataname = kwargs['dataname']
        eta = kwargs.pop('eta')
        epochs = kwargs.pop('epochs')
        datapath = P("datasets/data").joinpath(dataname)

        try:
            train_main(
                eta=eta / 100,
                datapath=datapath,
                batch_size=128,
                epochs=epochs,
                lr=0.0001,

                lambdas=1,
                alpha=0.01,
                temperature=0.005,
                train_size=0.9,


                savedir=savedir,
                save_history=True,
                save_vars=True,

                device='cpu',
                **kwargs
            )
        except:
            traceback.print_exc()


    Parallel(n_jobs=1, verbose=999)(
        delayed(func)(kwargs)
        for kwargs in kv_product(
            idx=range(1),
            eta=[10],
            epochs=[50],
            dataname=[
                "ORL-40.mat",
            ],

        )
    )
