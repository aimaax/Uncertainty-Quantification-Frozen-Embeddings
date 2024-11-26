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
    coco_valid_loader = loaders['val']

    # Load CLIP model
    CLIP_Net = load_model(device=device, model_path=None)

    # Define BayesCap network
    ProbVLM_Net = BayesCap_for_CLIP(
        inp_dim=512,
        out_dim=512,
        hid_dim=256,
        num_layers=3
    )

    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    ProbVLM_Net.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Evaluate using the existing `eval_ProbVLM` function
    mean_mae = eval_ProbVLM(
        CLIP_Net,
        ProbVLM_Net,
        coco_valid_loader,
        device=device
    )

    # Print or calculate additional statistics if needed
    print(f"Mean MAE from evaluation: {mean_mae}")

    # Return the evaluated metrics
    return mean_mae

def eval_ProbVLM_uncert(
    CLIP_Net,
    BayesCap_Net,
    eval_loader,
    device='cuda',
    n_fw=15,  # Number of forward passes for uncertainty estimation
    bins_type='eq_samples',  # 'eq_spacing' or 'eq_samples'
    n_bins=5,  # Number of uncertainty bins
):
    CLIP_Net.to(device)
    CLIP_Net.eval()
    BayesCap_Net.to(device)
    BayesCap_Net.eval()

    # Get features and uncertainties
    r_dict = get_features_uncer_ProbVLM(CLIP_Net, BayesCap_Net, eval_loader)

    # Sort samples by uncertainty
    sort_v_idx, sort_t_idx = sort_wrt_uncer(r_dict)

    # Bin the sorted samples
    if bins_type == 'eq_spacing':
        bins = create_uncer_bins_eq_spacing(sort_v_idx, n_bins=n_bins)
    elif bins_type == 'eq_samples':
        bins = create_uncer_bins_eq_samples(sort_v_idx, n_bins=n_bins)
    else:
        raise ValueError("Invalid `bins_type`. Choose 'eq_spacing' or 'eq_samples'.")

    # Calculate recall@1 for each bin
    bin_recalls = []
    counter = 0

    # For saving features of the samples to produce visualized PCA from 512 dim to 2
    PCA_query_features = np.empty((0, 512))
    PCA_gallery_features = np.empty((0, 512))

    for bin_key, samples in bins.items():
        if not samples:
            bin_recalls.append(0)  # If bin is empty, append 0
            continue
        
        indices = [sample[0] for sample in samples]  # Extract indices from sorted list
        bin_query_features = torch.stack([r_dict['ir_f'][i] for i in indices])
        bin_gallery_features = torch.stack(r_dict['t_f'])  # All gallery features
        # append PCA lists
        PCA_query_features = np.concatenate((PCA_query_features, bin_query_features.cpu().numpy()), axis=0)
        np.save('/mimer/NOBACKUP/groups/ulio_inverse/UQ/Uncertainty-Quantification-Frozen-Embeddings/alvis_code/ProbVLM/src/PCA_query_features.npy', PCA_query_features)
        pred_ranks = get_pred_ranks(bin_query_features, bin_gallery_features, recall_ks=(1,))
        
        if counter == 0:
            #print(f"Query: {bin_query_features}")
            #print(f"Gallery: {bin_gallery_features}")
            print(f"Pred Ranks: {pred_ranks}")
            #print(f"Indices: {indices}")
        counter += 1
        recall_scores = get_recall_COCOFLICKR(pred_ranks, recall_ks=(1,), q_idx=indices)
        bin_recalls.append(recall_scores[0])

    PCA_gallery_features = np.concatenate((PCA_gallery_features, bin_gallery_features.cpu().numpy()), axis=0)
    np.save("/mimer/NOBACKUP/groups/ulio_inverse/UQ/Uncertainty-Quantification-Frozen-Embeddings/alvis_code/ProbVLM/src/PCA_gallery_features.npy", PCA_gallery_features)
    return bin_recalls, bins


def load_and_evaluate_uncert(
    ckpt_path='../ckpt/ProbVLM_Net_best.pth',
    dataset="coco",
    data_dir="../datasets/coco",
    batch_size=64,
    device='cuda',
    n_bins=5,
    bins_type='eq_samples',
):
    # Load data loaders
    dataloader_config = mch({
        "batch_size": batch_size,
        "random_erasing_prob": 0,
        "traindata_shuffle": True
    })

    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    coco_valid_loader = loaders['val']

    # Load CLIP model
    CLIP_Net = load_model(device=device, model_path=None)

    # Define BayesCap network
    ProbVLM_Net = BayesCap_for_CLIP(
        inp_dim=512,
        out_dim=512,
        hid_dim=256,
        num_layers=3
    )

    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    ProbVLM_Net.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Evaluate with uncertainty
    bin_recalls, bins = eval_ProbVLM_uncert(
        CLIP_Net,
        ProbVLM_Net,
        coco_valid_loader,
        device=device,
        n_bins=n_bins,
        bins_type=bins_type
    )

    # Plot recall@1 vs uncertainty bins
    bin_labels = [f'Bin {i+1}' for i in range(len(bin_recalls))]
    plt.figure(figsize=(10, 6))
    plt.plot(bin_labels, bin_recalls, color='skyblue')
    plt.xlabel('Uncertainty Bins')
    plt.ylabel('Recall@1')
    plt.title('Recall@1 vs. Binned Uncertainty Levels')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("recall_plot.png")

    return bin_recalls, bins

def main():
    model = "../ckpt/BBB_woKL_Net_best.pth"
    #eval_results = load_and_evaluate()
    #print(eval_results)
    bin_recalls, bins = load_and_evaluate_uncert(ckpt_path=model)
    print("Recall@1 for each bin:", bin_recalls)

    #iod_ood = compare_iod_vs_ood(ckpt_path=model)


if __name__ == "__main__":
    main()

