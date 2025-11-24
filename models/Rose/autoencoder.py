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

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from einops import rearrange
from munkres import Munkres
from sklearn import metrics

import math
from typing import List
from utils.torch_utils import *

#from ..deepcca.run_dcca import batch_size



class Con_encoder(nn.Module):
    def __init__(self, input_dim, architecture:list[int],feature_dim,):
        super(Con_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, architecture[0]),
            nn.ReLU(),
            nn.Linear(architecture[0], architecture[1]),
            nn.ReLU(),
            nn.Linear(architecture[1], architecture[2]),
            nn.ReLU(),
            nn.Linear(architecture[2], feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Con_Decoder(nn.Module):
    def __init__(self, input_dim,architecture:list[int], feature_dim):
        super(Con_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, architecture[2]),  # 改
            nn.ReLU(),
            nn.Linear(architecture[2], architecture[1]),  # 改
            nn.ReLU(),
            nn.Linear(architecture[1], architecture[0]),
            nn.ReLU(),
            nn.Linear(architecture[0], input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class Spe_Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Spe_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),  # 改
            nn.ReLU(),
            nn.Linear(512, feature_dim),  # 改
        )

    def forward(self, x):
        return self.encoder(x)


class Spe_Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Spe_Decoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),  # 改
            nn.ReLU(),
            nn.Linear(1024, input_dim),  # 改
        )

    def forward(self, x):
        return self.encoder(x)





class Con_MultiviewEncoder(nn.Module):
    """
    The multi-view encoder part of the model.
    """

    def __init__(self, hidden_dim: int,
                 c_dim:int,
                 input_dims: List[int],
                 architecture,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.c_dim= c_dim
        self.input_dims=input_dims
        self.architecture=architecture

        self.viewNum = len(input_dims)


        self.fc_mu_single = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.c_dim))
                                          for _ in range(self.viewNum)])
        self.fc_logvar_single = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.c_dim))
                                           for _ in range(self.viewNum)])

        self.gating = nn.ModuleList([nn.Sequential(nn.Linear(self.c_dim, self.c_dim),
                                                   nn.ReLU(),
                                                   nn.Linear(self.c_dim, 1)
                                                   )
                                     for _ in range(self.viewNum)])

        self.to_encoder_input = nn.Linear(self.c_dim, self.hidden_dim)



    def Con_reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)  # mean 0, std
        return eps.mul(std).add_(mu)



    def forward(self, inputs: dict):

        X_views: List[Tensor] = inputs["X_views"]

        min_samples = min([x.shape[0] for x in X_views])
        
        mu_views = []* self.viewNum
        logvar_views = []* self.viewNum
        gate_weights = []* self.viewNum

        for v in range(self.viewNum):
            # Encoding
            self.Con_Encoder = Con_encoder(input_dim=self.input_dims[v], architecture=self.architecture,
                                           feature_dim=self.hidden_dim)

            X_view = X_views[v][:min_samples]
            H_view = self.Con_Encoder(X_view)

            # Fusion
            mu_view = self.fc_mu_single[v](H_view)
            logvar_view = self.fc_logvar_single[v](H_view)

            gate_weights.append(self.gating[v](H_view))

            mu_views.append(mu_view)
            logvar_views.append(logvar_view)

        mu_fusion = torch.stack(mu_views, dim=1)  # [views, batch, c_dim]
        logvar_fusion = torch.stack(logvar_views, dim=1)
        weights = F.softmax(torch.stack(gate_weights, dim=1), dim=1)

        mu = (weights * mu_fusion).sum(dim=1)
        logvar = (weights * logvar_fusion).sum(dim=1)

        H_common=self.Con_reparameterize(mu,logvar)



        inputs["mu_views"] = mu_views
        inputs["logvar_views"] = logvar_views
        inputs["H_common"] = H_common

        return inputs


class Con_MultiviewDecoder(nn.Module):
    """
    The decoder for view completion, i.e., completion-pretraining stage.
    """

    def __init__(self, hidden_dim: int,architecture:list[int],
                 c_dim:int,input_dims: List[int],):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder_views = nn.ModuleList()
        self.c_dim=c_dim

        for input_dim in input_dims:
            decoder = Con_Decoder(input_dim=input_dim,architecture=architecture, feature_dim=self.hidden_dim)
            self.decoder_views.append(decoder)

        self.to_decoder_input = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, inputs: dict):

        H:Tensor = inputs["H_common"]
        X_hats = [None] * len(self.decoder_views)
        for v in range(len(self.decoder_views)):
            X_hats[v] = self.decoder_views[v](H)
        inputs["X_hats"] = X_hats
        return inputs


class Spe_MultiviewEncoder(nn.Module):
    def __init__(self,device, hidden_dim: int,
                 input_dims: List[int],
                 number_components:int,
                 s_dim: int,
                 ):
        super().__init__()
        self.device=device
        self.hidden_dim = hidden_dim
        self.input_dims=input_dims
        self.number_components=number_components
        self.s_dim=s_dim
        self.pseudoinputs_mean =0.5
        self.pseudoinputs_std=0.01


        self.means = nn.ModuleList()
        self.encoder_view = nn.ModuleList()
        self.viewNum = len(input_dims)
        for input_dim in input_dims:
            encoder = Spe_Encoder(
                input_dim=input_dim, feature_dim=self.hidden_dim
            )

            self.encoder_view.append(encoder)


        self.to_dist_layer = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.s_dim * 2))
                                          for _ in range(self.viewNum)])

        self.fc_mu_single = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.s_dim))
                                           for _ in range(self.viewNum)])
        self.fc_logvar_single = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.s_dim))
                                               for _ in range(self.viewNum)])


        self.to_encoder_input = nn.Linear(self.s_dim, self.hidden_dim)

        self.add_pseudoinputs()


    def add_pseudoinputs(self):

        for v in range(self.viewNum):
            nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
            mean = NonLinear(self.number_components, self.input_dims[v], bias=False, activation=nonlinearity)
            self.normal_init(mean.linear, self.pseudoinputs_mean, self.pseudoinputs_std)
            self.means.append(mean)

        # create an idle input for calling pseudo-inputs
        self.idle_input = torch.eye(self.number_components, self.number_components, requires_grad=False).to(self.device)

    def normal_init(self,m, mean=0.0, std=0.01):
        m.weight.data.normal_(mean, std)


    def init_pseudoinput(self, all_samples, device):

        for v in range(self.viewNum):
            view_sample = all_samples[v]
            samples = torch.stack(view_sample)
            n = samples.shape[0]
            samples = samples.view(n, -1)
            samples = samples.to(device)

            self.means[v].linear.weight.data = samples.T

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(logvar)
        eps = torch.randn_like(std)  # mean 0, std
        return eps.mul(std).add_(mu)


    def forward(self, X_views: List[Tensor], inputs=None):

        if inputs is None:
            inputs = {}
            
        Pri_views=[]* self.viewNum
        Pos_views=[]* self.viewNum
        H_views=[]* self.viewNum
        

        min_samples = min([x.shape[0] for x in X_views])
        for v in range(self.viewNum):

            x = X_views[v][:min_samples]
            latent = self.encoder_view[v](x)

            mu_v = self.fc_mu_single[v](latent)
            logvar_v = self.fc_logvar_single[v](latent)
            h_viev = self.reparameterize(mu_v, logvar_v)

            H_views.append(h_viev)

            # Pos_viev
            Pos_view = self.log_Normal_diag(h_viev, mu_v, logvar_v, dim=1)
            Pos_views.append(Pos_view)


            #Prior_s
            K = self.number_components

            Pseudo_input_view = self.means[v](self.idle_input)
            Pseudo_input_view = self.encoder_view[v](Pseudo_input_view)
            Pseudo_input_view = self.to_dist_layer[v](Pseudo_input_view)
            mu_X, logvar_X = torch.split(Pseudo_input_view, self.s_dim, dim=1)

            h_expand = h_viev.unsqueeze(1)
            means = mu_X.unsqueeze(0)
            logvars = logvar_X.unsqueeze(0)

            a = self.log_Normal_diag(h_expand, means, logvars, dim=2) - math.log(K)  # MB x C
            a_max, _ = torch.max(a, 1)

            # calculte log-sum-exp
            Prior_view = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
            Pri_views.append(Prior_view)


        inputs["H_views"] = H_views
        inputs["Pri_views"] = Pri_views
        inputs["Pos_views"] = Pos_views

        return inputs


    def log_Normal_diag(self,x, mean, log_var, average=True, dim=None):
        log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
        if average:
            return torch.mean(log_normal, dim)
        else:
            return torch.sum(log_normal, dim)


class Spe_MultiviewDecoder(nn.Module):
    """
    The decoder for view completion, i.e., completion-pretraining stage.
    """

    def __init__(self, hidden_dim: int,s_dim, input_dims: List[int],):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.s_dim=s_dim
        self.decoder_view = nn.ModuleList()
        for input_dim in input_dims:
            decoder = Spe_Decoder(input_dim=input_dim, feature_dim=self.hidden_dim)
            self.decoder_view.append(decoder)

        self.to_decoder_input = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, inputs: dict):

        H_views:Tensor = inputs["H_views"]
        X_hats = [None] * len(self.decoder_view)
        for v in range(len(self.decoder_view)):
            h_view = self.to_decoder_input(H_views[v])
            X_hats[v] = self.decoder_view[v](h_view)
        inputs["X_hats"] = X_hats
        return inputs


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        Pseudo_input = self.activation(h)

        return Pseudo_input