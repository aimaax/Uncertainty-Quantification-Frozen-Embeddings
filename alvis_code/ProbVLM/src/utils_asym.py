import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
from tqdm import tqdm_notebook
import numpy as np
import torch
import torch.nn as nn

import clip
import ds 
from ds import prepare_coco_dataloaders, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders
from tqdm import tqdm
from losses import *

import numpy as np

def load_data_loader(dataset, data_dir, dataloader_config):
    prepare_loaders = {
        'coco': prepare_coco_dataloaders,
        'flickr': prepare_flickr_dataloaders,
        'CUB':prepare_cub_dataloaders,
        'FLO':prepare_flo_dataloaders
    }[dataset]
    if dataset == 'CUB':
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            caption_root=data_dir+'/text_c10',
            vocab_path='ds/vocabs/cub_vocab.pkl')
    elif dataset == 'FLO':
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            caption_root=data_dir+'/text_c10',)
    else:
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            vocab_path="/mimer/NOBACKUP/groups/ulio_inverse/UQ/Uncertainty-Quantification-Frozen-Embeddings/alvis_code/ProbVLM/src/ds/vocabs/coco_vocab.pkl")

    return loaders

def load_model(device, model_path=None):
    # load zero-shot CLIP model
    model, _ = clip.load(name='ViT-B/32',
                         device=device,
                         loss_type='contrastive')
    if model_path is None:
        # Convert the dtype of parameters from float16 to float32
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    else:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model

   
def multi_fwpass_ProbVLM(
    BayesCap_Net,
    xfI, xfT,
    n_fw=15
):
    txt_mu, txt_sigma = [], []
    BayesCap_Net.eval()

    for i in range(n_fw):
        (txt_mu, txt_sigma) = BayesCap_Net(xfT)
        txt_mu.append(txt_mu)
        txt_sigma.append(txt_sigma)
    ##
    t_mu = torch.cat(txt_mu, dim=0)
    t_sigma = torch.cat(txt_sigma, dim=0)
    t_mu_m, t_mu_v = torch.mean(t_mu, dim=0), torch.var(t_mu, dim=0)
    return (t_mu_m, t_mu_v)
 
def get_features_uncer_ProbVLM(
    CLIP_Net,
    BayesCap_Net,
    t_loader,
    device='cuda'
):
    r_dict= {
        'i_f': [],
        't_f': [],
        'ir_f': [],
        'tr_f': [],
        'i_au':[],
        'i_eu':[],
        'i_u': [],
        't_au':[],
        't_eu':[],
        't_u': [],
        'classes': []
    }
    # extract all features
    with torch.no_grad():
        for i_inputs, t_inputs, class_labels, _ in tqdm(t_loader):
            r_dict['classes'].extend(class_labels.cpu().tolist())
            n_batch = i_inputs.shape[0]
            i_inputs, t_inputs = i_inputs.to(device), t_inputs.to(device)
            outputs = CLIP_Net(i_inputs, t_inputs)
            #recons
            outs = multi_fwpass_ProbVLM(BayesCap_Net, outputs[0], outputs[1])
            # (i_mu_m, i_alpha_m, i_beta_m, i_v), (t_mu_m, t_alpha_m, t_beta_m, t_v)
            for j in range(n_batch):
                r_dict['i_f'].append(outputs[0][j,:])
                r_dict['t_f'].append(outputs[1][j,:])
                r_dict['ir_f'].append(outs[0][0][j,:])
                r_dict['tr_f'].append(outs[1][0][j,:])
                u = get_GGuncer(1/outs[0][1][j,:], outs[0][2][j,:])
                #print("aleatoric i: ", u)
                r_dict['i_au'].append(u)
                r_dict['i_eu'].append(outs[0][3][j,:])
                r_dict['i_u'].append(u) # outs[0][3][j,:]) # only aleatoric
                u = get_GGuncer(1/outs[1][1][j,:], outs[1][2][j,:])
                #print("aleatoric t: ", u)
                r_dict['t_au'].append(u)
                r_dict['t_eu'].append(outs[1][3][j,:])
                r_dict['t_u'].append(u) # outs[1][3][j,:]) # only aleatoric
    
    return r_dict

