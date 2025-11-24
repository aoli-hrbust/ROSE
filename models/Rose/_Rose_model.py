import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .autoencoder import *
from typing import List
#from utils.clustering_layer import *
from utils.metrics import *


class Rose_NetModel(nn.Module):

    def __init__(self,device, architecture, input_dims: List[int],
                 clusterNum: int, hidden_dim: int,
                 number_components:int,
                 c_dim:int,
                 s_dim:int):
        super(Rose_NetModel, self).__init__()
        assert clusterNum == architecture[-1]
        self.device=device
        self.viewNum = len(input_dims)
        self.hidden_dim = hidden_dim
        self.clusterNum=clusterNum
        self.number_components=number_components
        self.c_dim=c_dim
        self.s_dim=s_dim

        self.weights = nn.Parameter(
            torch.full((self.viewNum,), 1 / self.viewNum),
            requires_grad=True
        )


        self.Con_encoder = Con_MultiviewEncoder(hidden_dim=self.hidden_dim,c_dim=self.c_dim,
                                                input_dims=input_dims,architecture=architecture)

        self.Con_decoder = Con_MultiviewDecoder(hidden_dim=self.hidden_dim,architecture=architecture,
                                                c_dim=self.c_dim,input_dims=input_dims)

        self.Spe_encoder = Spe_MultiviewEncoder(device=self.device,hidden_dim=self.hidden_dim,input_dims=input_dims,
                                               number_components= self.number_components,s_dim=self.s_dim)

        self.Spe_decoder = Spe_MultiviewDecoder(hidden_dim=self.hidden_dim,s_dim=self.s_dim,input_dims=input_dims)

        self.centers = nn.Parameter(torch.empty(self.clusterNum, self.clusterNum))
        nn.init.normal_(self.centers.data)




    def forward(
                self, X: List[torch.Tensor],
                M: torch.Tensor,
                y:torch.Tensor

        ):

            actual_view_num = min(self.viewNum, len(X))


            M = M.to(torch.bool)

            X_views = [X[v][M[:, v]] for v in range(actual_view_num)]



            inputs = dict(X_views=X_views)
            inputs = self.Con_encoder(inputs)
            mu_views_Con = inputs['mu_views']
            logvar_views_Con = inputs['logvar_views']
            H_common = inputs['H_common']


            inputs = self.Con_decoder(inputs)
            X_hats_Con = inputs['X_hats']



            # Spe_EncoderNetwork
            inputs=self.Spe_encoder(X_views)
            H_views=inputs["H_views"]
            Pri_views=inputs["Pri_views"]
            Pos_views=inputs["Pos_views"]

            # Spe_decoderNetwork
            inputs=self.Spe_decoder(inputs)
            X_hats_Spe = inputs['X_hats']


            alpha=1
            Q_views=[]*self.viewNum

            centers, _ = KMeans_Torch(H_common, n_clusters=self.clusterNum)
            self.centers.data = centers

            for v in range(self.viewNum):

                Q_view = 1.0 / (1.0 + torch.sum(torch.pow(H_views[v].unsqueeze(1) - self.centers, 2), 2) / alpha)
                Q_view = Q_view.pow((alpha + 1.0) / 2.0)
                Q_view = (Q_view.t() / torch.sum(Q_view, 1)).t()
                Q_views.append(Q_view)

            # Fusion(inputs:H_views)
            Q_common= self.fusion(Q_views,M)

            P_common = self.get_P_common(H_common, self.centers)


            outputs = dict(X_views=X_views, mu_views_Con=mu_views_Con, logvar_views_Con =logvar_views_Con,
                            P_common=P_common,X_hats_Con=X_hats_Con,
                           Q_views=Q_views,Pri_views=Pri_views,Pos_views=Pos_views,X_hats_Spe=X_hats_Spe,
                           Q_common=Q_common
                           )

            return outputs



    def get_weights(self):
        softmax_weights = torch.softmax(self.weights, dim=0)
        weights = softmax_weights / torch.sum(softmax_weights)

        return weights

    def fusion(self, Q_views,M):
        # Fusion
        SampleNum = min([Q_view.shape[0] for Q_view in Q_views])
        weights = self.get_weights()

        Q_common = torch.zeros(SampleNum, self.clusterNum).to(M.device)

        for v in range(self.viewNum):

            Q_common+= weights[v] * Q_views[v]
        Q_common = F.normalize(Q_common / self.viewNum)

        return Q_common

    def get_P_common(self,h_common,centers,alpha=1):

        Q_common = 1.0 / (1.0 + torch.sum(torch.pow(h_common.unsqueeze(1) - centers, 2), 2) / alpha)
        Q_common = Q_common.pow((alpha + 1.0) / 2.0)
        Q_common = (Q_common.t() / torch.sum(Q_common, 1)).t()
        weight = Q_common ** 2 / Q_common.sum(0)
        P_common=(weight.t() / weight.sum(1)).t().detach()

        return P_common




