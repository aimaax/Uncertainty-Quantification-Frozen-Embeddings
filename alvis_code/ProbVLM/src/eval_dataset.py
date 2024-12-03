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
import time 

import clip
import ds 
from ds import prepare_coco_dataloaders
from tqdm import tqdm
from losses import *
#from utils import *
from utils_implementation import * 

from ds.vocab import Vocabulary

from networks import *
from networks_mc_do import *
from networks_BBB_EncBL import *

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
    elif model_type == "BBB_EncBL":
        Net = BayesCap_for_CLIP_BBB_Enc(
            inp_dim = 512,
            out_dim = 512,
            hid_dim = 256,
            num_layers = 3
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
    n_fw=15,
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
    elif model_type == "BBB_EncBL":
        Net = BayesCap_for_CLIP_BBB_Enc(
            inp_dim = 512,
            out_dim = 512,
            hid_dim = 256,
            num_layers = 3
        )
    else:
        print(f"Error: Unknown model type '{model}'. Supported models are 'ProbVLM' and 'BBB'.", file=sys.stderr)
        sys.exit(1)

    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    Net.load_state_dict(torch.load(ckpt_path, map_location=device))

    CLIP_Net.to(device)
    CLIP_Net.eval()
    Net.to(device)
    Net.eval()

    print("Calculating uncertainties...")
    uncert_dict = get_uncertainties(CLIP_Net, Net, valid_loader, n_fw=n_fw, device="cuda") 
    recall_values, bins = compute_recall_for_uncertainty_bins(uncert_dict, recall_k=1, n_bins=5)

    # Plot the recall vs uncertainty
    plot_recall_vs_uncertainty(recall_values, n_bins=5)
    
    """
    summary = summarize_uncertainties(uncert_dict)

    for key, stats in summary.items():
        print(f"{key}:")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.4f}")
    """

    return (0, 0)


def main():
    #model = "../ckpt/BBB_EncBL_best_first.pth"
    model = "../ckpt/BBB_EncBL_best_3.pth"
    #model = "../ckpt/ProbVLM_Net_best.pth"
    #model = "../ckpt/ProbVLM_Net_best_3.pth"
    
    start = time.time()
    print("Evaluating with load_and_evaluate...")
    #mae_coco_probvlm = load_and_evaluate(ckpt_path=model, dataset="coco", data_dir="../datasets/coco", model_type="ProbVLM")
    mae_flickr_probvlm = load_and_evaluate(ckpt_path=model, batch_size=1, dataset="flickr", data_dir="../datasets/flickr", model_type="BBB_EncBL")
    #mae_flickr_BBB_Enc = load_and_evaluate(ckpt_path=model, dataset="flickr", data_dir="../datasets/flickr", model_type="BBB_EncBL")
    print(f"Completed in {time.time() - start:.2f} seconds.")
    
    start = time.time()
    print("Estimating uncertainty with uncert_est...")
    #uncert_coco_BBB = uncert_est(ckpt_path=model, dataset="coco", data_dir="../datasets/coco", model_type="BBB_EncBL", n_fw=50)
    #uncert_flicker_BBB = uncert_est(ckpt_path=model, dataset="flickr", data_dir="../datasets/flickr", model_type="BBB_EncBL")
    #uncert_coco_probvlm = uncert_est(ckpt_path=model, dataset="coco", data_dir="../datasets/coco", model_type="ProbVLM")
    uncert_flickr_probvlm = uncert_est(ckpt_path=model, batch_size=1, dataset="flickr", data_dir="../datasets/flickr", model_type="BBB_EncBL")
    print(f"Completed in {time.time() - start:.2f} seconds.")
if __name__ == "__main__":
    main()

