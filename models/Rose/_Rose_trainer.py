import logging
import random
import torch.distributed as dist
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,  TensorDataset
from sklearn.neighbors import kneighbors_graph
from tqdm import trange, tqdm
from ._utils import *

# from ._trainer import Trainer
from ._Rose_loss import Rose_NetLoss
from ._Rose_model import Rose_NetModel

from utils import convert_tensor
from utils.metrics import *

import math
#from utils.clustering_layer import *


class Rose_Trainer:
    def __init__(
            self,
            input_dims: List[int],
            clusterNum: int,
            config: dict,
            device: torch.device,
            architecture: list[int],
    ):


        self.device = device
        self.architecture = architecture
        self.input_dims = input_dims
        self.clusterNum=clusterNum
        self.ViewNum = len(input_dims)
        self.config = config



        self.lr = self.config["lr"]
        self.missing_rate = self.config["eta"]
        self.epochs = self.config["epochs"]
        self.batch_size = self.config["batch_size"]
        self.hidden_dim = self.config["hidden_dim"]


        self.c_dim=self.config["c_dim"]
        self.s_dim=self.config["s_dim"]
        self.scheduler=self.config["scheduler"]
        self.pseudoinputs_number=self.config["pseudoinputs_number"]
        self.number_components=self.config["number_components"]
        self.pseudoinputs_training_data_init=True
        self.seed=3407



        self.alpha = self.config["alpha"]
        self.beta = self.config["beta"]


        self.Con_Loss = self.config["Con_Loss"]
        self.Spe_Loss = self.config["Spe_Loss"]
        self.Cur_Loss = self.config["Cur_Loss"]
        self.Pse_Loss = self.config["Pse_Loss"]

        

        self.epoch = 0

        self.Rose_net = Rose_NetModel(device=self.device,
            architecture=self.architecture,
            input_dims=self.input_dims,
            clusterNum=self.clusterNum,
            hidden_dim=self.hidden_dim,
            number_components=self.number_components,
            c_dim=self.c_dim,
            s_dim=self.s_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.Rose_net.parameters(), lr=self.lr)

        self.criterion = Rose_NetLoss(alpha=self.alpha,beta=self.beta,
            ViewNum=self.ViewNum,scheduler=self.scheduler,
            Con_Loss=self.Con_Loss, Spe_Loss=self.Spe_Loss,Cur_Loss =self.Cur_Loss,
            Pse_Loss=self.Pse_Loss,
            epoch=self.epoch, epochs=self.epochs,
            ClusterNum=self.clusterNum

          )

        self.mm = MaxMetrics()

    def train(self, X: List[torch.Tensor], y: torch.Tensor, M: torch.Tensor):


        self.X = convert_tensor(X)
        self.y = convert_tensor(y, dtype=torch.long)
        self.M = convert_tensor(M, dtype=torch.bool)
        self.counter = 0

        best_outputs = None

        self.history = []

        train_loader, test_loader = self._get_data_loader()


        #itialize the pseudo-input parameters
        if self.pseudoinputs_number != 0 and self.pseudoinputs_training_data_init:

            n_samples = self.X[0].shape[0]
            indices = torch.randperm(n_samples)[:self.pseudoinputs_number]
            if len(indices) < self.pseudoinputs_number:
                cup = self.pseudoinputs_number - len(indices)
                indices = torch.cat((indices, indices[:cup]))


            all_samples = [[self.X[i][j] for j in indices] for i in range(self.ViewNum)]
            self.init_pseudoinputs(all_samples, self.device)


        print("Training Rose_Net:")
        t = trange(self.epochs, leave=True)
        for epoch in t:
            self.epoch=epoch

            Loss_all=0.0
            Cur_loss = 0.0
            pse_loss = 0.0


            for (*X_grad, M_grad,y_grad) in train_loader:

                X_grad = [x.to(self.device) for x in X_grad]
                M_grad = M_grad.to(self.device)
                y_grad = y_grad.to(self.device)

                # Gradient step
                self.Rose_net.train()
                self.optimizer.zero_grad()

                outputs = self.Rose_net(X_grad, M_grad,y_grad)
                outputs["M"] = M_grad

                loss = self.criterion(outputs)
                loss["Loss_all"].backward()
                self.optimizer.step()



                Loss_all += loss["Loss_all"]
                Cur_loss += loss["Cur_loss"].item()
                pse_loss += loss["pse_loss"].item()


            # End of batch. Begin epoch's evaluation.
            Loss_all /= len(train_loader)
            Cur_loss /= len(train_loader)
            pse_loss /= len(train_loader)


            Q_common = self.predict(test_loader)
            metrics = Rose_get_all_metrics(y, convert_numpy(Q_common.argmax(1)))

            metrics.update(loss=Loss_all)

            if self.mm.update(**metrics)["ACC"]:
                best_outputs = convert_numpy(
                    dict(Q_common=Q_common)
                )
            self.history.append(metrics)


            t.set_description(
                "Loss_all: {:.5f}\n"
                "Cur_loss: {:.5f}, pse_loss: {:.5f}\n"
                "ACC: {:.2f}, NMI: {:.2f}, PUR: {:.2f}".format(
                Loss_all,
                    Cur_loss, pse_loss,
                    metrics["ACC"] , metrics["NMI"], metrics["PUR"]
                )
            )
            t.refresh()

        return self.mm.report(current=False), best_outputs




    def init_pseudoinputs(self,all_samples, device ):

        self.Rose_net.Spe_encoder.init_pseudoinput(all_samples,device)


    def _get_data_loader(self) -> tuple:
        """
        This function returns the data loaders for training, validation and testing.

        Returns:
            tuple:  The data loaders
        """
        n = self.X[0].shape[0]
        if self.y is None:
            self.y = torch.zeros(n)

        dataset = TensorDataset(*self.X, self.M, self.y)
        train_dataset = dataset
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader



    @torch.no_grad()
    def predict(self, test_loader):
        """Predicts the cluster assignments for the given data.

        Parameters
        ----------
        X : torch.Tensor
            Data to be clustered.

        Returns
        -------
        np.ndarray
            The cluster assignments for the given data.
        """

        Q_common_list = []

        for index, (*X, M,y_grad) in enumerate(test_loader):
            # for *X, M in test_loader:
            X = [x.to(self.device) for x in X]
            M = M.to(self.device)
            y_grad = y_grad.to(self.device)

            outputs = self.Rose_net(X, M,y_grad)
            q_common=outputs["Q_common"]
            Q_common_list.append(q_common)


        Q_common = torch.cat(Q_common_list, 0)
        Q_common = F.normalize(Q_common)

        return  Q_common


