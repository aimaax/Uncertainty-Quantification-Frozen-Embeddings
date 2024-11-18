import os
from os.path import join as ospj
from os.path import expanduser

import numpy as np
import torch
import torch.nn as nn
import torchbnn as bnn

import clip
from tqdm import tqdm
from utils_lavis import *


class BayesCap_MLP(nn.Module): # Edited
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
        prior_mu=0.5,
        prior_sigma=0.1
    ):
        super(BayesCap_MLP, self).__init__()
        mod = []
        for layer in range(num_layers):
            if layer==0:
                incoming = inp_dim
                outgoing = hid_dim
                mod.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                    in_features=incoming, out_features=outgoing))
                mod.append(nn.ReLU())
            elif layer==num_layers//2:
                incoming = hid_dim
                outgoing = hid_dim
                mod.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                    in_features=incoming, out_features=outgoing))
                mod.append(nn.ReLU())
            elif layer==num_layers-1:
                incoming = hid_dim
                outgoing = out_dim
                mod.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                    in_features=incoming, out_features=outgoing))
        self.mod = nn.Sequential(*mod)

        self.block_mu = nn.Sequential(
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                in_features=out_dim, out_features=out_dim),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                in_features=out_dim, out_features=out_dim),
        )

        self.block_alpha = nn.Sequential(
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                in_features=out_dim, out_features=out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                in_features=out_dim, out_features=out_dim),
            nn.ReLU(),
        )

        self.block_beta = nn.Sequential(
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                in_features=out_dim, out_features=out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                in_features=out_dim, out_features=out_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x_intr = self.mod(x)
        # print('dbg', x_intr.shape, x.shape)
        x_intr = x_intr + x
        x_mu = self.block_mu(x_intr)
        x_1alpha = self.block_alpha(x_intr)
        x_beta = self.block_beta(x_intr)
        return x_mu, x_1alpha, x_beta




class BayesCLIP(nn.Module):
    def __init__(
        self,
        model_path=None,
        device='cuda',
    ):
        super(BayesCLIP, self).__init__()
        self.clip_model = load_model(device, model_path)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.img_BayesCap = BayesCap_MLP(inp_dim=512, out_dim=512, hid_dim=512, num_layers=3, prior_mu=0.5, prior_sigma=0.1).to(device)
        self.txt_BayesCap = BayesCap_MLP(inp_dim=512, out_dim=512, hid_dim=512, num_layers=3, prior_mu=0.5, prior_sigma=0.1).to(device)

    def forward(self, i_inputs, t_inputs):
        i_features, t_features = self.clip_model(i_inputs, t_inputs)

        img_mu, img_1alpha, img_beta = self.img_BayesCap(i_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(t_features)

        return (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta), (i_features, t_features)


class BayesCap_for_CLIP(nn.Module): # Edited 
    def __init__(
        self,
        inp_dim=512,
        out_dim=512,
        hid_dim=256,
        num_layers=3,
        prior_mu=0.5,
        prior_sigma=0.,
    ):
        super(BayesCap_for_CLIP, self).__init__()
        self.img_BayesCap = BayesCap_MLP(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.txt_BayesCap = BayesCap_MLP(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, prior_mu=prior_mu, prior_sigma=prior_sigma)

    def forward(self, i_features, t_features):
        
        # print('dbg', i_features.shape, t_features.shape)
        img_mu, img_1alpha, img_beta = self.img_BayesCap(i_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(t_features)

        return (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta)

