import math
from typing import List, Any, Union
import itertools

import torch
import torch.nn as nn
from sympy import discriminant
#from torch.nn.functional import cross_entropy

from utils.metrics import *
from utils.torch_utils import *
from datasets.get_mask import get_mask



class Rose_NetLoss(nn.Module):
    def __init__(self, alpha=0.02, beta=0.02,
                 ViewNum=None,scheduler=None,
                 epoch=None, ClusterNum=None,
                 epochs=None,
                 Con_Loss: bool = True,
                 Spe_Loss: bool = True,
                 Cur_Loss: bool = True,
                 Pse_Loss: bool = True,
                 ):
        super(Rose_NetLoss, self).__init__()

        self.ViewNum=ViewNum
        self.ClusterNum=ClusterNum
        self.scheduler=scheduler
        self.epoch=epoch
        self.epochs=epochs

        self.alpha = alpha
        self.beta=beta


        self.Con_loss = ConsistentViewLoss(ViewNum=self.ViewNum,alpha=self.alpha)
        self.Spe_loss = SpecificViewLoss(ViewNum=self.ViewNum)
        self.Pse_loss = Pseudo_supervisedLoss(clusterNum=self.ClusterNum)


        self.Con_Loss = Con_Loss
        self.Spe_Loss = Spe_Loss
        self.Cur_Loss = Cur_Loss
        self.Pse_Loss = Pse_Loss

    def forward(self, outputs: dict):

        if self.Con_Loss:

            KL_loss_con, Recon_loss_con, DA_loss,con_loss  = self.Con_loss(outputs)
        else:
            con_loss=0.0

        if self.Spe_Loss:
            #Spe_loss
            KL_loss_spe, Recon_loss_spe,spe_loss=self.Spe_loss(outputs)
        else:
            spe_loss=0.0

        if self.Cur_Loss:
            #Cur_loss
            lambda_ = self.gen_lambda(self.epoch,self.epochs)
            Cur_loss = lambda_ * con_loss + (1 - lambda_) * spe_loss
        else:
            Cur_loss=0.0

        if self.Pse_Loss:
            #Pse_loss
            pse_loss=self.Pse_loss(outputs)
        else:
            pse_loss=0.0



        Loss_all = Cur_loss + self.beta * pse_loss
        loss = dict( Cur_loss=Cur_loss, pse_loss=pse_loss,
            Loss_all=Loss_all
        )

        return loss

    def gen_lambda(self, epoch: int, epochs:int, alpha=2.0) -> Union[float, None, Any]:
        """
        Generate the adaptive parameter according to the current epoch.

        Parameters
        ----------
        epoch : int
            The current epoch.

        Returns
        -------
        float
            The trade-off parameter.
        """
        T_T_max = (epoch + 1) / epochs

        if self.scheduler == "linear":
            return 1.0 - T_T_max
        elif self.scheduler == "exponential":
            return math.exp(-2 * math.pi * T_T_max)
        elif self.scheduler == "cosine":
            return math.cos(T_T_max * math.pi) / 2.0 + 0.5
        elif self.scheduler == "power":
            return 1.0 - T_T_max ** alpha
        else:
            assert False, f"Scheduler type [{self.scheduler}] not in pre-defined."



class ConsistentViewLoss(nn.Module):
    """
    MSE-based view completion loss
    """

    def __init__(self, ViewNum: [int], alpha):
        super().__init__()
        self.ViewNum = ViewNum
        self.alpha=alpha
        self.kld_weight=1.0

        # Losses
        self.l2_norm = nn.MSELoss(reduction='sum')
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.Kld_loss = nn.KLDivLoss(reduction='sum')

    def forward(self, inputs: dict):
        mu_views_Con: list[torch.Tensor] = inputs["mu_views_Con"]
        logvar_views_Con: list[torch.Tensor] = inputs["logvar_views_Con"]

        X_views: list[torch.Tensor] = inputs["X_views"]
        X_hats_Con: list[torch.Tensor] = inputs["X_hats_Con"]

        KL_view = 0
        recon_loss = 0
        da_loss=0
        for v in range(self.ViewNum):

            #VAE_loss
            KL = torch.mean(-0.5 * torch.sum(1 + logvar_views_Con[v] - mu_views_Con[v] ** 2 - logvar_views_Con[v].exp(),
                                             dim=1), dim=0)

            KL_view += self.kld_weight * KL

            min_samples = min([x.shape[0] for x in X_views])
            X_view = X_views[v][:min_samples]
            recon_loss += F.mse_loss(X_view, X_hats_Con[v], reduction='mean')

            #AD_Loss

            if not mu_views_Con or not logvar_views_Con:
                raise ValueError("输入列表不能为空")

            if len(logvar_views_Con) != len(logvar_views_Con):
                raise ValueError("均值列表和对数方差列表长度必须相同")


            if self.ViewNum == 1:
                return torch.tensor(0.0, device=mu_views_Con[0].device)

            sigma_list = [logvar.exp() for logvar in logvar_views_Con]


            other_mu_sum = sum(mu_views_Con[i] for i in range(self.ViewNum) if i != v)
            other_sigma_sum = sum(sigma_list[i] for i in range(self.ViewNum) if i != v)

            avg_other_mu = other_mu_sum / (self.ViewNum - 1)
            avg_other_sigma = other_sigma_sum / (self.ViewNum - 1)


            mu_diff = self.l2_norm(mu_views_Con[v], avg_other_mu)
            sigma_diff = torch.norm(sigma_list[v] - avg_other_sigma)

            da_loss += mu_diff + sigma_diff


        KL_loss_con = KL_view / self.ViewNum
        Recon_loss_con = recon_loss / self.ViewNum
        da_loss_con = da_loss/self.ViewNum

        VAE_loss=Recon_loss_con+KL_loss_con
        DA_loss=torch.sqrt(da_loss_con)

        Con_loss = VAE_loss + self.alpha*DA_loss

        return KL_loss_con, Recon_loss_con, DA_loss,   Con_loss


class SpecificViewLoss(nn.Module):
    """
    MSE-based view completion loss
    """
    def __init__(self, ViewNum:[int]):
        super().__init__()
        self.ViewNum = ViewNum
        self.kld_weight=1.0

        self.recons_criterion = nn.MSELoss(reduction='mean')


    def forward(self, inputs: dict):
        Pri_views:list[torch.Tensor]=inputs["Pri_views"]
        Pos_views:list[torch.Tensor]=inputs["Pos_views"]

        X_views:list[torch.Tensor]=inputs["X_views"]
        X_hats_Spe:list[torch.Tensor]=inputs["X_hats_Spe"]


        KL_view = 0
        recon_loss=0
        for v in range(self.ViewNum):

            KL = -(Pri_views[v] - Pos_views[v])

            KL_view += self.kld_weight * torch.mean(KL)

            min_samples = min([x.shape[0] for x in X_views])
            X_view = X_views[v][:min_samples]
            recon_loss += self.recons_criterion(X_view,X_hats_Spe[v])


        KL_loss_spe = KL_view / self.ViewNum
        Recon_loss_spe= recon_loss / self.ViewNum

        Spe_loss=Recon_loss_spe+KL_loss_spe

        return  KL_loss_spe, Recon_loss_spe,     Spe_loss


class Pseudo_supervisedLoss(nn.Module):
    """
    t-SNE based manifold regularization loss.
    """
    def __init__(self,clusterNum):
        super().__init__()
        self.centers = nn.Parameter(torch.empty(clusterNum, clusterNum))  # nn.Parameter 是 PyTorch 中的一个类,用于将一个张量标记为需要学习的参数。
        nn.init.normal_(self.centers.data)  # nn.init.normal_ 是 PyTorch 中的一个初始化函数,它会将输入的张量用均值为 0、标准差为 1 的正态分布进行初始化。

    def forward(self, inputs: dict):
        Q_views: List[Tensor] = inputs["Q_views"]
        P_common: Tensor = inputs["P_common"]
        viewNum: int = len(Q_views)


        kl_loss = 0
        for v in range(viewNum):

            kl_loss += F.kl_div(Q_views[v].log(), P_common, reduction="batchmean")

        Pse_loss = kl_loss / viewNum
        return Pse_loss



    def get_Q_view(self, H_view, mu, alpha=1):
        """
        z: embeddings; mu: centroids
        q_{iu} = P(cluster(i)==u|i)
        """
        Q = 1.0 / (1.0 + torch.sum(torch.pow(H_view.unsqueeze(1) - mu, 2), 2) / alpha)
        Q = Q.pow((alpha + 1.0) / 2.0)
        Q_view = (Q.t() / torch.sum(Q, 1)).t()
        return Q_view
