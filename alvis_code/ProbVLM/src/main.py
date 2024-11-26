import sys
import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
from tqdm import tqdm_notebook
import numpy as np
import torch
import torch.nn as nn
import time

import clip
import ds 
from ds import prepare_coco_dataloaders
from tqdm import tqdm
from losses import *
from utils import *

from ds.vocab import Vocabulary

from networks import *
from networks_mc_do import *
from uncertainty_estimates import *

import torchbnn as bnn

def train_ProbVLM(
    CLIP_Net,
    BayesCap_Net,
    train_loader,
    eval_loader,
    model,
    Cri = TempCombLoss(),
    device='cuda',
    dtype=torch.cuda.FloatTensor(),
    init_lr=1e-4,
    num_epochs=100,
    eval_every=1,
    ckpt_path='../ckpt/ProbVLM',
    cross_modal_lambda=1e-4,
    T1=1e0,
    T2=5e-2
):
    CLIP_Net.to(device)
    CLIP_Net.eval()
    ##
    BayesCap_Net.to(device)
    BayesCap_Net.img_BayesCap.train()
    BayesCap_Net.txt_BayesCap.train()
    ##
    optimizer = torch.optim.Adam(
        list(BayesCap_Net.img_BayesCap.parameters())+list(BayesCap_Net.txt_BayesCap.parameters()), 
        lr=init_lr
    )
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # early stopping and adaptive learning rate
    patience = 8
    early_stop_counter = 0

    # KL divergence for BBB loss
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    M = len(train_loader) 

    metrics = {
        "train_loss": [],
        "val_mse": [],
        "val_mae": [],
        "epochs": []
    }

    score = 1e8
    start_time = time.time()

    for eph in range(num_epochs):
        eph_loss = 0
        BayesCap_Net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for (idx, batch) in enumerate(tepoch):
                #if idx>500:
                #    break
                tepoch.set_description('Epoch {}'.format(eph))
                ##
                xI, xT  = batch[0].to(device), batch[1].to(device)
                with torch.no_grad():
                    xfI, xfT = CLIP_Net(xI, xT)
                # xI, xT = xI.type(dtype), xT.type(dtype)
                # pass them through the network
                (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)
                
                optimizer.zero_grad()
                loss_i = Cri(img_mu, img_1alpha, img_beta, xfI, T1=T1, T2=T2)
                loss_t = Cri(txt_mu, txt_1alpha, txt_beta, xfT, T1=T1, T2=T2)
                #cross modal terms
                loss_i4t = Cri(img_mu, img_1alpha, img_beta, xfT, T1=T1, T2=T2)
                loss_t4i = Cri(txt_mu, txt_1alpha, txt_beta, xfI, T1=T1, T2=T2)
                if model == "ProbVLM":
                    loss = loss_i + loss_t + cross_modal_lambda*(loss_i4t + loss_t4i)
                elif model == "BBB":
                    kl = kl_loss(BayesCap_Net)
                    kl_weight = 2**(M-idx-1)/(2**M-1) # pi_i weight according to paper
                    loss = loss_i + loss_t + cross_modal_lambda*(loss_i4t + loss_t4i) + kl_weight*kl
                # print(loss)
                loss.backward()
                optimizer.step()
                ##
                eph_loss += loss.item()
                metrics["train_loss"].append(eph_loss)
                tepoch.set_postfix(loss=loss.item())
            eph_loss /= len(train_loader)
            print('Avg. loss: {}'.format(eph_loss))

        # evaluate and save the models
        torch.save(BayesCap_Net.state_dict(), ckpt_path+'_last.pth')
        if eph%eval_every == 0:
            curr_score = eval_ProbVLM(
                CLIP_Net,
                BayesCap_Net,
                eval_loader,
                device=device,
                dtype=dtype,
            )
            print('current score: {} | Last best score: {}'.format(curr_score, score))
            metrics["val_mse"].append(val_mse)
            metrics["val_mae"].append(val_mae)
            metrics["epochs"].append(eph)
            if curr_score <= score:
                score = curr_score
                torch.save(BayesCap_Net.state_dict(), ckpt_path+'_best.pth')
                early_stop_counter = 0
                best_epoch = eph
            else: 
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {eph}. Best score: {score} at epoch {best_epoch}.")
                break
        optim_scheduler.step()
    
    end_time = time.time()
    print("Total training time: ", end_time - start_time)

    np.save(ckpt_path + "_metrics.npy", metrics)
    print(f"Metrics saved to {ckpt_path}_metrics.npy")


def main(model):
    dataset = "coco"
    data_dir = ospj("../datasets/", dataset)
    dataloader_config = mch({
        "batch_size":128,
        "random_erasing_prob":0,
        "traindata_shuffle":True
    })

    #loaders, vocab = load_data_loader(dataset, data_dir, dataloader_config)
    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    coco_train_loader, coco_valid_loader, coco_test_loader = loaders['train'], loaders['val'], loaders['test']

    CLIP_Net = load_model(device='cuda', model_path = None)
    if model == "ProbVLM":
        Net = BayesCap_for_CLIP_ProbVLM(
                inp_dim = 512,
                out_dim = 512,
                hid_dim = 256,
                num_layers=3
        )

        model_path = "../ckpt/ProbVLM_Net"
    elif model == "BBB":
        Net = BayesCap_for_CLIP(
                inp_dim = 512,
                out_dim = 512,
                hid_dim = 256,
                num_layers=3
        )
        
        model_path = "../ckpt/BBB_Net"
    else:
        # Print an error message and exit the program
        print(f"Error: Unknown model type '{model}'. Supported models are 'ProbVLM' and 'BBB'.", file=sys.stderr)
        sys.exit(1) 

    print(f"Training model {model}...")

    train_ProbVLM(
            CLIP_Net,
            Net,
            coco_train_loader,
            coco_valid_loader,
            model,
            Cri = TempCombLoss(),
            device='cuda',
            dtype=torch.cuda.FloatTensor,
            init_lr=8e-5,
            num_epochs=500,
            eval_every=1,
            ckpt_path = model_path,
            T1=1e0,
            T2=1e-4
    )


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "ProbVLM"
    main(model)

