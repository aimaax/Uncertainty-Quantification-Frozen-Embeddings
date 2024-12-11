import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
import numpy as np

import torchbnn as bnn

class AsymLoss(nn.Module):
    def __init__(self) -> None:
        super(AsymLoss, self).__init__()

    def forward(self, txt_mu, log_var, img_mu):
        loss1 = ((txt_mu - img_mu) ** 2 / log_var.exp()).sum(dim=1) / 2
        loss2 = log_var.sum(dim=1) / 2
        return (loss1 + loss2).mean()

