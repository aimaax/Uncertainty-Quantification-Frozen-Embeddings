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
from losses_asym import *
from utils import *

from ds.vocab import Vocabulary

from networks_asym import *
from uncertainty_estimates_asym import *

import torchbnn as bnn

def train_ProbVLM(
    CLIP_Net,
    BayesCap_Net,
    train_loader,
    eval_loader,
    model,
    device='cuda',
    dtype=torch.cuda.FloatTensor(),
    init_lr=1e-4,
    num_epochs=15,
    eval_every=1,
    ckpt_path='../ckpt/ProbVLM',
    cross_modal_lambda=1e-4,
    T1=1e0,
    T2=1e-4
):
    CLIP_Net.to(device)
    CLIP_Net.eval()
    BayesCap_Net.to(device)
    BayesCap_Net.txt_BayesCap.train()
    optimizer = torch.optim.Adam(
        list(BayesCap_Net.txt_BayesCap.parameters()), 
        lr=init_lr
    )
    #optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    patience = 100
    early_stop_counter = 0
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 1e-5)
    kl_loss = bnn.BKLLoss(reduction='sum', last_layer_only=False) # normalized
    M = len(train_loader) # number of matches
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
        "val_mae": [],
        "epochs": [],
        "kl": []
    }

    parameters = {
            "txt_mu": [],
            "txt_log_variance": [],
            "img_mu": []
    }

    score = 1e8
    start_time = time.time()
    
    model_last_path = f"{ckpt_path}_last.pth"
    model_best_path = f"{ckpt_path}_best.pth"
    metrics_path = ckpt_path + "_metrics.npy"
    parameters_path = ckpt_path + "_parameters.npy"

    counter = 0
    if os.path.exists(model_last_path):
        counter = 1
        while os.path.exists(f"{ckpt_path}_last_{counter}.pth"):
            counter += 1
        model_last_path = f"{ckpt_path}_last_{counter}.pth"

    print("Model counter: ", counter)
    model_best_path = f"{ckpt_path}_best_{counter}.pth"
    metrics_path = f"{ckpt_path}_metrics_{counter}.npy"
    parameters_path = f"{ckpt_path}_parameters_{counter}.npy"
    loss_function = AsymLoss()

    for eph in range(num_epochs):
        eph_loss = 0
        BayesCap_Net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            (txt_mu, txt_log_variance) = (0, 0)
            for (idx, batch) in enumerate(tepoch):
                tepoch.set_description('Epoch {}'.format(eph))
                xI, xT  = batch[0].to(device), batch[1].to(device)
                with torch.no_grad():
                    #xfI = model_CLIP.encode_image(xI)
                    #xfT = model_CLIP.encode_text(xT)
                    xfI, xfT = CLIP_Net(xI, xT)
                (txt_mu, txt_log_variance, img_mu) = BayesCap_Net(xfI, xfT)

                optimizer.zero_grad()
                kl_loss_img = kl_loss(BayesCap_Net)
                kl_lambda = 1
                loss_NLL = loss_function(txt_mu, txt_log_variance, img_mu)
                loss = loss_NLL + kl_loss_img * kl_lambda
                loss.backward()


                optimizer.step()
                eph_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            eph_loss /= len(train_loader)
            print('Avg. loss per batch: {}'.format(eph_loss))
            metrics["train_loss"].append(eph_loss)
            parameters["txt_mu"].append(txt_mu)
            parameters["txt_log_variance"].append(txt_log_variance)
            parameters["img_mu"].append(img_mu)
        
        # Print learning rate
        print(f"Epoch {eph}: Learning rate is {optimizer.param_groups[0]['lr']:.6f}")
        print(f"txt_log_variance: {txt_log_variance}")
        print(f"kl_loss: {kl_loss_img}, kl_lambda: {kl_lambda}")
        print(f"loss_NLL: {loss_NLL}")

        # evaluate and save the models
        torch.save(BayesCap_Net.state_dict(), model_last_path)
        if eph%eval_every == 0:
            val_mae, val_mse, val_loss = eval_ProbVLM(
                CLIP_Net,
                BayesCap_Net,
                eval_loader,
                device=device,
                dtype=dtype,
            )
            #optim_scheduler.step(val_mae)
            print('current val_loss score: {} | Last best val_loss score: {}'.format(val_loss, score))
            #if model == "BBB" or model == "BBB_EncBL" or model=="asym":
            #    #print(f"kl: {kl} | kl_weight: {kl_weight}")
            #    metrics["kl"].append(kl_loss_img)
            metrics["val_loss"].append(val_loss)
            metrics["val_mse"].append(val_mse)
            metrics["val_mae"].append(val_mae)
            metrics["epochs"].append(eph)

            if val_loss <= score:
                score = val_loss
                torch.save(BayesCap_Net.state_dict(), model_best_path)
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

    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    coco_train_loader, coco_valid_loader, coco_test_loader = loaders['train'], loaders['val'], loaders['test']

    CLIP_Net = load_model(device='cuda', model_path = None)
    Net = BayesCap_for_CLIP_asym(
            inp_dim = 512,
            out_dim = 512,
            hid_dim = 256,
            num_layers=3
    )

    model_path = "../ckpt/ASYM_BBB"

    print(f"Training model {model}...")

    train_ProbVLM(
            CLIP_Net,
            Net,
            coco_train_loader,
            coco_valid_loader,
            model,
            device='cuda',
            dtype=torch.cuda.FloatTensor,
            #init_lr=8e-5,
            init_lr=1e-4,
            num_epochs=100,
            eval_every=1,
            ckpt_path = model_path,
            T1=1,
            T2=1e-4
    )

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "ProbVLM"
    main(model)


