import os
from os.path import join as ospj
from os.path import expanduser

import numpy as np
import torch
import torch.nn as nn
import torchbnn as bnn

import clip
from tqdm import tqdm
from utils import *


class BayesCap_MLP_img(nn.Module): 
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: int, Input dimension
        out_dim: int, Output dimension
        hid_dim: int, hidden dimension
        num_layers: Number of hidden layers
        prior_mu: Mean for layer
        prior_sigma: Standard deviation for layer
    '''
    def __init__(
        self, 
        inp_dim, 
        out_dim,
        hid_dim=512, 
        num_layers=1, 
        prior_mu=0.0,
        prior_sigma=1.0
    ):
        super(BayesCap_MLP_img, self).__init__()

        mod_img = []
        for layer in range(num_layers):
            if layer==0:
                incoming = inp_dim
                outgoing = hid_dim
                mod_img.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                    in_features=incoming, out_features=outgoing))
                mod_img.append(nn.ReLU())
            elif layer==num_layers//2:
                incoming = hid_dim
                outgoing = hid_dim
                mod_img.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                    in_features=incoming, out_features=outgoing))
                mod_img.append(nn.ReLU())
            elif layer==num_layers-1:
                incoming = hid_dim
                outgoing = out_dim
                mod_img.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                    in_features=incoming, out_features=outgoing))
        self.mod_img = nn.Sequential(*mod_img)

        self.block_mu = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        
    def forward(self, x):
        x_mod = self.mod_img(x)
        mu = self.block_mu(x_mod)
        return mu


class BayesCap_MLP_txt(nn.Module): 
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: int, Input dimension
        out_dim: int, Output dimension
        hid_dim: int, hidden dimension
        num_layers: Number of hidden layers
        prior_mu: Mean for layer
        prior_sigma: Standard deviation for layer
    '''
    def __init__(
        self, 
        inp_dim, 
        out_dim,
        hid_dim=512, 
        num_layers=1, 
        prior_mu=0.0,
        prior_sigma=1.0
    ):
        super(BayesCap_MLP_txt, self).__init__()
        mod_txt = []
        for layer in range(num_layers):
            if layer==0:
                incoming = inp_dim
                outgoing = hid_dim
                mod_txt.append(nn.Linear(out_dim, out_dim))
                mod_txt.append(nn.ReLU())
            elif layer==num_layers//2:
                incoming = hid_dim
                outgoing = hid_dim
                mod_txt.append(nn.Linear(out_dim, out_dim))
                mod_txt.append(nn.ReLU())
            elif layer==num_layers-1:
                incoming = hid_dim
                outgoing = out_dim
                mod_txt.append(nn.Linear(out_dim, out_dim))

        self.mod_txt = nn.Sequential(*mod_txt)

        self.block_log_variance = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    
    def forward(self, x):
        x_mod = self.mod_txt(x)
        log_var = self.block_log_variance(x_mod) 
        return log_var


class BayesCap_for_CLIP_asym(nn.Module): 
    def __init__(
        self,
        inp_dim=512,
        out_dim=512,
        hid_dim=256,
        num_layers=3,
        prior_mu=0.0,
        prior_sigma=1.0,
    ):
        super(BayesCap_for_CLIP_asym, self).__init__()
        self.txt_BayesCap = BayesCap_MLP_txt(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.img_BayesCap = BayesCap_MLP_img(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, prior_mu=prior_mu, prior_sigma=prior_sigma)

    def forward(self, i_features, t_features):
        """
        Returns xfI + noise (where noise is trained)
        """
        if t_features is not None:
            txt_mu = t_features
            txt_log_variance = self.txt_BayesCap(t_features)
        else:
            txt_mu, txt_log_variance = None, None

        if i_features is not None:
            #img_mu = i_features + self.img_BayesCap(i_features)
            img_mu = self.img_BayesCap(i_features)
        else:
            img_mu = None

        return (txt_mu, txt_log_variance, img_mu)


