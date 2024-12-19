import torch
from torch import nn
import torchbnn as bnn
import numpy as np

class AsymProbAdaptor(nn.Module):
    def __init__(self):
        super(AsymProbAdaptor, self).__init__()
        mod_txt = []
        mod_txt.append(nn.Linear(512, 512))
        mod_txt.append(nn.BatchNorm1d(512))
        mod_txt.append(nn.ReLU())
        mod_txt.append(nn.Linear(512, 512))
        mod_txt.append(nn.BatchNorm1d(512))
        mod_txt.append(nn.ReLU())
        mod_txt.append(nn.Linear(512, 512))

        self.mod_txt = nn.Sequential(*mod_txt)

        self.txt_log_var = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        mod_img = []
        mod_img.append(nn.Linear(512, 512))
        mod_img.append(nn.BatchNorm1d(512))
        mod_img.append(nn.ReLU())
        mod_img.append(nn.Linear(512, 512))
        mod_img.append(nn.BatchNorm1d(512))
        mod_img.append(nn.Dropout(p=0.3))
        mod_img.append(nn.ReLU())
        mod_img.append(nn.Linear(512, 512))

        self.mod_img = nn.Sequential(*mod_img)

        self.img_mu = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, xfT=None, xfI=None):
        if xfT is not None:
            txt_mu = xfT
            txt_log_var = self.txt_log_var(self.mod_txt(xfT))
        else:
            txt_mu, txt_log_var = None, None
        if xfI is not None:
            xfI = xfI.half()
            img_mu = xfI + self.img_mu(self.mod_img(xfI))
        else:
            img_mu = None
        return txt_mu, txt_log_var, img_mu

    def loss(self, txt_mu, txt_log_var, img_mu, image_features):
        image_features = image_features.repeat(5, 1)
        img_mu = img_mu.repeat(5, 1)
        loss1 = ((txt_mu - img_mu) ** 2 / txt_log_var.exp()).sum(dim=1) / 2
        loss2 = txt_log_var.sum(dim=1) / 2
        loss3 = ((img_mu - image_features) ** 2).sum(dim=1) / 2
        lambda_loss3 = 0.5

        return (loss1 + loss2 + lambda_loss3 * loss3).mean()


class BBBLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_var=1.0):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var
        self.prior_mu = prior_mu

        self.W_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_log_var = nn.Parameter(torch.Tensor(out_features, in_features))

        self.b_mu = nn.Parameter(torch.Tensor(out_features))
        self.b_log_var = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W_mu)
        nn.init.constant_(self.W_log_var, -3)
        nn.init.zeros_(self.b_mu)
        nn.init.constant_(self.b_log_var, -3)

    def forward(self, x):
        epsilon_W = torch.randn(self.W_mu.size(), device=self.W_mu.device)
        epsilon_b = torch.randn(self.b_mu.size(), device=self.b_mu.device)

        W = self.W_mu + self.W_log_var.exp() * epsilon_W
        b = self.b_mu + self.b_log_var.exp() * epsilon_b

        return nn.functional.linear(x, W, b)

    def kl_divergence(self):
        w_var = self.W_log_var.exp().float()
        b_var = self.b_log_var.exp().float()

        kl_w = 0.5 * (w_var.sum() + (self.W_mu - self.prior_mu).pow(2).sum() \
            - w_var.log().sum() - np.log(self.prior_var) * self.W_mu.numel() - self.W_mu.numel())
        kl_b = 0.5 * (b_var.sum() + (self.b_mu - self.prior_mu).pow(2).sum() \
            - b_var.log().sum() - np.log(self.prior_var) * self.b_mu.numel() - self.b_mu.numel())

        return (kl_w + kl_b)

class AsymProbAdaptorBNN(nn.Module):
    def __init__(self, prior_mu=0.0, prior_sigma=1.0):
        super(AsymProbAdaptorBNN, self).__init__()
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        mod_txt = []
        mod_txt.append(nn.Linear(512, 512))
        mod_txt.append(nn.BatchNorm1d(512))
        mod_txt.append(nn.ReLU())
        mod_txt.append(nn.Linear(512, 512))
        mod_txt.append(nn.BatchNorm1d(512))
        mod_txt.append(nn.ReLU())
        mod_txt.append(nn.Linear(512, 512))

        self.mod_txt = nn.Sequential(*mod_txt)

        self.txt_log_var = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        mod_img = []
        mod_img.append(BBBLinear(512, 512))
        mod_img.append(nn.BatchNorm1d(512))
        mod_img.append(nn.ReLU())
        mod_img.append(BBBLinear(512, 512))
        mod_img.append(nn.BatchNorm1d(512))
        mod_img.append(nn.ReLU())
        mod_img.append(BBBLinear(512, 512))

        self.mod_img = nn.Sequential(*mod_img)

        self.img_mu = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, xfT=None, xfI=None):
        if xfT is not None:
            txt_mu = xfT
            txt_log_var = self.txt_log_var(self.mod_txt(xfT))
        else:
            txt_mu, txt_log_var = None, None
        if xfI is not None:
            img_mu = xfI + self.img_mu(self.mod_img(xfI))
        else:
            img_mu = None
        return txt_mu, txt_log_var, img_mu

    def loss(self, txt_mu, txt_log_var, img_mu, image_features):
        image_features = image_features.repeat(5, 1)
        img_mu = img_mu.repeat(5, 1)
        
        # NLL term
        loss1 = ((txt_mu - img_mu) ** 2 / txt_log_var.exp()).sum(dim=1) / 2
        loss2 = txt_log_var.sum(dim=1) / 2

        # Regularization term
        loss3 = ((img_mu - image_features) ** 2).sum(dim=1) / 2
        lambda_loss3 = 0.5

        # KL term
        num_params = sum(p.numel() for p in self.parameters())
        kl = self.kl_divergence()

        return (loss1 + loss2 + lambda_loss3 * loss3).mean() + kl / num_params, kl / num_params

    def kl_divergence(self):
        kl = 0.0
        for layer in self.mod_img:
            if isinstance(layer, BBBLinear):
                kl += layer.kl_divergence()
        return kl

