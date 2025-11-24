"""
Curriculum learning: Efficient Multi-view Representation for
 Incomplete Multi-view Clustering with Pseudo-supervised(ROSE)
"""
import logging
from utils.torch_utils import *
from utils.io_utils import *
from ._Rose_trainer import Rose_Trainer
from datasets.dataset import *
from utils.metrics import *


def train_main(
        datapath=P("./data/ORL-40.mat"),
        eta=0.5,
        views=None,
        lr: float = 0.0001,
        epochs: int = 50,
        batch_size: int = 1024,

        Con_Loss: bool = True,
        Spe_Loss: bool = True,
        Cur_Loss: bool = True,
        Pse_Loss: bool = True,

        device=get_device(),

        savedir: P = P("datasets/data"),
        save_vars: bool = False,
        save_history: bool = False,
        train_size: float = 0.9,

        alpha=0.5,
        number_components=300,
        beta=0.02,

        c_dim=128,
        s_dim=128,
        hidden_dim=128,
        scheduler= "power",
        pseudoinputs_number=300,


        **kwargs,
):
    if train_size < 0.5:
        raise ValueError(f'train_size too small: {train_size}')

    method = "data"

    config = dict(
        datapath=datapath,
        eta=eta,
        views=views,
        method=method,
        batch_size=batch_size,
        lr=lr,

        epochs=epochs,
        device=device,

        train_size=train_size,
        hidden_dim=hidden_dim,
        c_dim=c_dim,
        s_dim=s_dim,
        scheduler=scheduler,
        pseudoinputs_number=pseudoinputs_number,

        alpha=alpha,
        number_components=number_components,
        beta=beta,


        Con_Loss=Con_Loss,
        Spe_Loss=Spe_Loss,
        Cur_Loss=Cur_Loss,
        Pse_Loss=Pse_Loss,

    )

    train_begin(savedir, config, f"Begin train {method}")


    data = PartialMultiviewDataset(
        datapath=datapath,
        paired_rate=1 - eta,
        view_ids=views,
        normalize="center",
    )


    architecture: list = [1024, 1024, 512, data.clusterNum]
    trainer = Rose_Trainer(
        input_dims=data.view_dims,
        clusterNum=data.clusterNum,
        config=config, device=device, architecture=architecture
    )

    begin = time.time()
    metrics, outputs = trainer.train(X=data.X, y=data.Y, M=data.mask)
    T = time.time() - begin
    metrics["T"] = T


    history = trainer.history

    if save_vars:
        save_variables(savedir, outputs)

    if save_history:
        save_var(savedir, history, "history")

    train_end(savedir, metrics, method)
