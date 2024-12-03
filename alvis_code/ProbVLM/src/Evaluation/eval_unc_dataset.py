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
from utils_implementation import * 

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

    CLIP_Net.to(device)
    CLIP_Net.eval()
    Net.to(device)
    Net.eval()

    print("Calculating uncertainties...")
    uncert_dict = get_uncertainties(CLIP_Net, Net, valid_loader, device="cuda") 
    
    summary = summarize_uncertainties(uncert_dict)
    for key, stats in summary.items():
        print(f"{key}:")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.4f}")

    i_au = torch.cat(uncert_dict["i_au"]).cpu().numpy()
    t_au = torch.cat(uncert_dict["t_au"]).cpu().numpy()
    i_eu = torch.cat(uncert_dict["i_eu"]).cpu().numpy()
    t_eu = torch.cat(uncert_dict["t_eu"]).cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(i_au, t_au, alpha=0.5, label="Aleatoric Uncertainty")
    plt.scatter(i_eu, t_eu, alpha=0.5, label="Epistemic Uncertainty")
    plt.title("Scatter Plot of Image vs. Text Uncertainties")
    plt.xlabel("Image Uncertainty")
    plt.ylabel("Text Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.savefig("uncert_test.png")
    
    """
    r_dict = get_features_uncer_ProbVLM(CLIP_Net, Net, valid_loader)

    i_eu_list = [elem for sublist in r_dict['i_eu'] for elem in sublist]
    i_au_list = [elem for sublist in r_dict['i_au'] for elem in sublist]
    t_eu_list = [elem for sublist in r_dict['t_eu'] for elem in sublist]
    t_au_list = [elem for sublist in r_dict['t_au'] for elem in sublist]
    avg_i_eu = sum(i_eu_list) / len(i_eu_list)
    avg_i_au = sum(i_au_list) / len(i_au_list)
    avg_t_eu = sum(t_eu_list) / len(t_eu_list)
    avg_t_au = sum(t_au_list) / len(t_au_list)

    print("\n*** Results ***")
    print(f"Avg. Image Epistemic Uncertainty: {avg_i_eu}, Avg. Text Epistemic Uncertainty: {avg_t_eu}")
    print(f"Avg. Image Aleatoric Uncertainty: {avg_i_au}, Avg. Text Aleatoric Uncertainty: {avg_t_au}")

    return (avg_i_eu, avg_t_eu)
    """
    return (0, 0)


def main():
    #model = "../ckpt/BBB_Net_best_first_KL.pth"
    model = "../ckpt/ProbVLM_Net_best.pth"
    #mae_coco_probvlm = load_and_evaluate(ckpt_path=model, dataset="coco", data_dir="../datasets/coco", model_type="BBB")
    #mae_flickr_probvlm = load_and_evaluate(ckpt_path=model, dataset="flickr", data_dir="../datasets/flickr", model_type="ProbVLM")
    #uncert_coco_BBB = uncert_est(ckpt_path=model, dataset="coco", data_dir="../datasets/coco", model_type="BBB")
    #uncert_flicker_BBB = uncert_est(ckpt_path=model, dataset="flickr", data_dir="../datasets/flickr", model_type="BBB")
    uncert_coco_probvlm = uncert_est(ckpt_path=model, dataset="coco", data_dir="../datasets/coco", model_type="ProbVLM")
    #uncert_flickr_probvlm = uncert_est(ckpt_path=model, dataset="flickr", data_dir="../datasets/flickr", model_type="ProbVLM")

if __name__ == "__main__":
    main()

