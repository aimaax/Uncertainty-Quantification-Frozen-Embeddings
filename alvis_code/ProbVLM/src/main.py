

"""




OBS::: BEFORE RUNNING, I CHANGED THE PATIENCE AND THE EPOCHS





"""

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
from networks_BBB_EncBL import *
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
    #optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    # early stopping and adaptive learning rate
    patience = 8
    early_stop_counter = 0

    # KL divergence for BBB loss
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    M = len(train_loader) # number of matches
    metrics = {
        "train_loss": [],
        "val_mse": [],
        "val_mae": [],
        "epochs": [],
        "kl": []
    }

    parameters = {
            "txt_mu": [],
            "txt_alpha": [],
            "txt_beta": [],
            "img_mu": [],
            "img_alpha": [],
            "img_beta": []
    }

    score = 1e8
    start_time = time.time()
    
    # model save path
    model_last_path = f"{ckpt_path}_last.pth"
    model_best_path = f"{ckpt_path}_best.pth"
    metrics_path = ckpt_path + "_metrics.npy"
    parameters_path = ckpt_path + "_parameters.npy"

    if os.path.exists(model_last_path):
        counter = 1
        while os.path.exists(f"{ckpt_path}_last_{counter}.pth"):
            counter += 1
        model_last_path = f"{ckpt_path}_last_{counter}.pth"


    if os.path.exists(model_best_path):
        counter = 1
        while os.path.exists(f"{ckpt_path}_best_{counter}.pth"):
            counter += 1
        model_best_path = f"{ckpt_path}_best_{counter}.pth"
    
    if os.path.exists(metrics_path):
        counter = 1
        while os.path.exists(f"{ckpt_path}_metrics_{counter}.npy"):
            counter += 1
        metrics_path = f"{ckpt_path}_metrics_{counter}.npy"

    if os.path.exists(parameters_path):
        counter = 1
        while os.path.exists(f"{ckpt_path}_parameters_{counter}.npy"):
            counter += 1
        parameters_path = f"{ckpt_path}_parameters_{counter}.npy"

    for eph in range(num_epochs):
        random_pi = np.random.rand(M)  # Sample random values from a uniform distribution
        pi = random_pi / random_pi.sum()  # Normalize to make the sum equal to 1
        pi = torch.tensor(pi, dtype=torch.float32, device=device)  # Move to GPU/CPU
        eph_loss = 0
        BayesCap_Net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            (img_mu, img_alpha, img_beta), (txt_mu, txt_alpha, txt_beta) = (0, 0, 0), (0, 0, 0)
            for (idx, batch) in enumerate(tepoch):
                # if idx>500:
                    # break
                tepoch.set_description('Epoch {}'.format(eph))
                ##
                xI, xT  = batch[0].to(device), batch[1].to(device)
                with torch.no_grad():
                    xfI, xfT = CLIP_Net(xI, xT)
                # xI, xT = xI.type(dtype), xT.type(dtype)
                # pass them through the network
                (img_mu, img_alpha, img_beta), (txt_mu, txt_alpha, txt_beta) = BayesCap_Net(xfI, xfT)
                
                optimizer.zero_grad()
                loss_i = Cri(img_mu, img_alpha, img_beta, xfI, T1=T1, T2=T2)
                loss_t = Cri(txt_mu, txt_alpha, txt_beta, xfT, T1=T1, T2=T2)
                #cross modal terms
                loss_i4t = Cri(img_mu, img_alpha, img_beta, xfT, T1=T1, T2=T2)
                loss_t4i = Cri(txt_mu, txt_alpha, txt_beta, xfI, T1=T1, T2=T2)
                if model == "ProbVLM":
                    loss = loss_i + loss_t + cross_modal_lambda*(loss_i4t + loss_t4i)
                elif model == "BBB" or model == "BBB_EncBL":
                    kl = kl_loss(BayesCap_Net)
                    #kl_weight = 2**(M-idx-1)/(2**M-1) # BBB paper scheme
                    #kl_weight = pi[idx]  # pi scheme, eq (9)
                    loss = loss_i + loss_t + cross_modal_lambda*(loss_i4t + loss_t4i)
                    if eph < 30:
                        desired_kl_ratio = 0.05
                    else:
                        desired_kl_ratio = 0.15
                    #kl_weight = desired_kl_ratio * loss / kl
                    kl_weight = 1/M
                    loss = loss_i + loss_t + cross_modal_lambda*(loss_i4t + loss_t4i) + kl_weight*kl
                # print(loss)
                loss.backward()
                optimizer.step()
                ##
                eph_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            eph_loss /= len(train_loader)
            print('Avg. loss per batch: {}'.format(eph_loss))
            metrics["train_loss"].append(eph_loss)
            parameters["txt_mu"].append(txt_mu)
            parameters["txt_alpha"].append(txt_alpha)
            parameters["txt_beta"].append(txt_beta)
            parameters["img_mu"].append(img_mu)
            parameters["img_alpha"].append(img_alpha)
            parameters["img_beta"].append(img_beta)
        
        # Print learning rate
        print(f"Epoch {eph}: Learning rate is {optimizer.param_groups[0]['lr']:.6f}")

        # evaluate and save the models
        torch.save(BayesCap_Net.state_dict(), model_last_path)
        if eph%eval_every == 0:
            val_mae, val_mse = eval_ProbVLM(
                CLIP_Net,
                BayesCap_Net,
                eval_loader,
                device=device,
                dtype=dtype,
            )
            optim_scheduler.step(val_mae)
            print('current mae score: {} | Last best mae score: {}'.format(val_mae, score))
            if model == "BBB" or model == "BBB_EncBL":
                print(f"kl: {kl} | kl_weight: {kl_weight}")
                metrics["kl"].append(kl)
            metrics["val_mse"].append(val_mse)
            metrics["val_mae"].append(val_mae)
            metrics["epochs"].append(eph)

            if val_mae <= score:
                score = val_mae
                torch.save(BayesCap_Net.state_dict(), model_best_path)
                early_stop_counter = 0
                best_epoch = eph
            else: 
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {eph}. Best score: {score} at epoch {best_epoch}.")
                break
        #optim_scheduler.step()
    
    end_time = time.time()
    print("Total training time: ", end_time - start_time)
    
    # save metrics and parameters 
    np.save(metrics_path, metrics)
    print(f"Metrics saved to {metrics_path}")

    np.save(parameters_path, parameters)
    print(f"Parameters saved to {parameters_path}")


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
    elif model == "BBB_EncBL":
        Net = BayesCap_for_CLIP_BBB_Enc(
                inp_dim = 512,
                out_dim = 512,
                hid_dim = 256,
                num_layers = 3
        )
        model_path = "../ckpt/BBB_EncBL"
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
            #init_lr=8e-5,
            init_lr=1e-4,
            num_epochs=100,
            eval_every=1,
            ckpt_path = model_path,
            T1=0,
            T2=1
    )

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "ProbVLM"
    main(model)

