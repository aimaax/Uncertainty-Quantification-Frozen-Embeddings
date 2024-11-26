import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
from tqdm import tqdm_notebook
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import utils

import clip
import ds 
from ds import prepare_coco_dataloaders
from tqdm import tqdm
from losses import *
from utils import *

from ds.vocab import Vocabulary

from networks import *
from networks_mc_do import *

def eval_ProbVLM(
    CLIP_Net,
    BayesCap_Net,
    eval_loader,
    device='cuda',
    dtype=torch.cuda.FloatTensor,
):
    CLIP_Net.to(device)
    CLIP_Net.eval()
    BayesCap_Net.to(device)
    BayesCap_Net.eval()

    mean_mse = 0
    mean_mae = 0
    num_imgs = 0
    list_error = []
    list_var = []
    with tqdm(eval_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):
            tepoch.set_description('Validating ...')
            ##
            xI, xT  = batch[0].to(device), batch[1].to(device)
            # xI, xT = xI.type(dtype), xT.type(dtype)
            
            # pass them through the network
            with torch.no_grad():
                xfI, xfT = CLIP_Net(xI, xT)
                (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)
                
            n_batch = img_mu.shape[0]
            for j in range(n_batch):
                num_imgs += 1
                mean_mse += emb_mse(img_mu[j], xfI[j]) + emb_mse(txt_mu[j], xfT[j])
                mean_mae += emb_mae(img_mu[j], xfI[j]) + emb_mae(txt_mu[j], xfT[j])
            ##
        mean_mse /= num_imgs
        mean_mae /= num_imgs
        print(
            'Avg. MSE: {} | Avg. MAE: {}'.format
            (
                mean_mse, mean_mae 
            )
        )
    return mean_mae

def load_and_evaluate(
    ckpt_path='../ckpt/ProbVLM_Net_best.pth',
    dataset="coco",
    data_dir="../datasets/coco",
    model_type="ProbVLM",
    batch_size=64,
    device='cuda'
    ):
    # Load data loaders
    dataloader_config = mch({
        "batch_size": batch_size,
        "random_erasing_prob": 0,
        "traindata_shuffle": True
    })

    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    valid_loader = loaders['val']

    # Load CLIP model
    CLIP_Net = load_model(device=device, model_path=None)

    if model_type == "BBB":
        # Define BayesCap network
        Net = BayesCap_for_CLIP(
            inp_dim=512,
            out_dim=512,
            hid_dim=256,
            num_layers=3
        )
    elif model_type == "ProbVLM":
        Net = BayesCap_for_CLIP_ProbVLM(
                inp_dim = 512,
                out_dim = 512,
                hid_dim = 256,
                num_layers=3
        )
    else:
        print(f"Error: Unknown model type '{model}'. Supported models are 'ProbVLM' and 'BBB'.", file=sys.stderr)
        sys.exit(1) 


    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    Net.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Evaluate using the existing `eval_ProbVLM` function
    mean_mae = eval_ProbVLM(
        CLIP_Net,
        Net,
        valid_loader,
        device=device
    )

    # Print or calculate additional statistics if needed
    print(f"Mean MAE from evaluation: {mean_mae}")

    # Return the evaluated metrics
    return mean_mae


def uncert_est(
    ckpt_path="../ckpt/ProbVLM_Net_best.pth",
    dataset="coco",
    data_dir="../datasets/coco",
    model_type="ProbVLM",
    batch_size=64,
    n_fw=10,
    device="cuda"
):
    # Load data loaders
    dataloader_config = mch({
        "batch_size": batch_size,
        "random_erasing_prob": 0,
        "traindata_shuffle": True
    })
    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    valid_loader = loaders['val']

    # Load CLIP model
    CLIP_Net = load_model(device=device, model_path=None)
    
    if model_type == "BBB":
        # Define BayesCap network
        Net = BayesCap_for_CLIP(
            inp_dim=512,
            out_dim=512,
            hid_dim=256,
            num_layers=3
        )
    elif model_type == "ProbVLM":
        Net = BayesCap_for_CLIP_ProbVLM(
            inp_dim=512,
            out_dim=512,
            hid_dim=256,
            num_layers=3
        )
    else:
        print(f"Error: Unknown model type '{model}'. Supported models are 'ProbVLM' and 'BBB'.", file=sys.stderr)
        sys.exit(1)

    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    Net.load_state_dict(torch.load(ckpt_path, map_location=device))

    image_uncertainties = []
    text_uncertainties = []

    CLIP_Net.to(device)
    CLIP_Net.eval()
    Net.to(device)
    Net.eval()

    with tqdm(valid_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):
            tepoch.set_description("Validating ...")
            xI, xT  = batch[0].to(device), batch[1].to(device)

            # Extract features using CLIP
            with torch.no_grad():
                xfI, xfT = CLIP_Net(xI, xT)

                # Compute image and text uncertainties
                (_, _, _, i_v), (_, _, _, t_v) = multi_fwpass_ProbVLM(
                    BayesCap_Net=Net,  
                    xfI=xfI,         
                    xfT=xfT,         
                    n_fw=n_fw,
                )
            if idx == 2:
                print(f"i_v: {i_v}")
                print(f"t_v: {t_v}")
                print(f"mean i_v: {i_v.mean().item()}")
                print(f"mean t_v: {t_v.mean().item()}")
                print(f"Max i_v: {i_v.max().item()}, Min i_v: {i_v.min().item()}")
                print(f"Max t_v: {t_v.max().item()}, Min t_v: {t_v.min().item()}")
                print(f"i_v shape: {i_v.shape}, t_v shape: {t_v.shape}")

                # Convert tensors to CPU for saving
                i_v_cpu = i_v.cpu().numpy()
                t_v_cpu = t_v.cpu().numpy()

                # Save i_v
                with open("i_v_batch_2.txt", "w") as f:
                    f.write("i_v Tensor:\n")
                    for row in i_v_cpu:
                        f.write(" ".join(map(str, row)) + "\n")

                # Save t_v
                with open("t_v_batch_2.txt", "w") as f:
                    f.write("t_v Tensor:\n")
                    for row in t_v_cpu:
                        f.write(" ".join(map(str, row)) + "\n")

                print("Tensors saved to 'i_v_batch_2.txt' and 't_v_batch_2.txt'")

            image_uncertainties.append(i_v.mean().item())
            text_uncertainties.append(t_v.mean().item())

    image_uncertainties_tensor = torch.tensor(image_uncertainties)
    avg_image_uncertainty = image_uncertainties_tensor.cpu().numpy().mean()
    text_uncertainties_tensor = torch.tensor(text_uncertainties)
    avg_text_uncertainty = text_uncertainties_tensor.cpu().numpy().mean()

    # Print results
    print("\n*** Results ***")
    print(f"Avg. Image Uncertainty: {avg_image_uncertainty}, Avg. Text Uncertainty: {avg_text_uncertainty}")

    return (avg_image_uncertainty, avg_text_uncertainty)


def main():
    #model = "../ckpt/BBB_woKL_Net_best.pth"
    model = "../ckpt/ProbVLM_Net_best.pth"
    #mae_coco_probvlm = load_and_evaluate(ckpt_path=model, dataset="coco", data_dir="../datasets/coco", model_type="ProbVLM")
    #mae_flickr_probvlm = load_and_evaluate(ckpt_path=model, dataset="flickr", data_dir="../datasets/flickr", model_type="ProbVLM")
    uncert_coco_probvlm = uncert_est(ckpt_path=model, dataset="coco", data_dir="../datasets/coco", model_type="ProbVLM")

if __name__ == "__main__":
    main()

