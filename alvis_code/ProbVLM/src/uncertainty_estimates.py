import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
from tqdm import tqdm_notebook
import numpy as np
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

def eval_ProbVLM_with_uncertainty(
    CLIP_Net,
    BayesCap_Net,
    eval_loader,
    device='cuda',
    dtype=torch.cuda.FloatTensor,
    n_bins=10,
    recall_ks=(1, 5, 10),
):
    CLIP_Net.to(device)
    CLIP_Net.eval()
    BayesCap_Net.to(device)
    BayesCap_Net.eval()

    mean_mse = 0
    mean_mae = 0
    num_imgs = 0

    all_features = {'i_f': [], 't_f': [], 'ir_f': [], 'tr_f': [], 'i_u': [], 't_u': []}
    all_classes = []

    with tqdm(eval_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):
            tepoch.set_description('Validating ...')
            
            xI, xT, class_labels = batch[0].to(device), batch[1].to(device), batch[2].cpu().tolist()
            with torch.no_grad():
                xfI, xfT = CLIP_Net(xI, xT)
                (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)
                
            n_batch = img_mu.shape[0]
            for j in range(n_batch):
                num_imgs += 1
                mean_mse += emb_mse(img_mu[j], xfI[j]) + emb_mse(txt_mu[j], xfT[j])
                mean_mae += emb_mae(img_mu[j], xfI[j]) + emb_mae(txt_mu[j], xfT[j])

                # Store features and uncertainty
                all_features['i_f'].append(xfI[j].cpu())
                all_features['t_f'].append(xfT[j].cpu())
                all_features['ir_f'].append(img_mu[j].cpu())
                all_features['tr_f'].append(txt_mu[j].cpu())
                i_unc = get_GGuncer(img_1alpha[j], img_beta[j])
                t_unc = get_GGuncer(txt_1alpha[j], txt_beta[j])
                all_features['i_u'].append(i_unc.cpu())
                all_features['t_u'].append(t_unc.cpu())
                all_classes.append(class_labels[j])

        mean_mse /= num_imgs
        mean_mae /= num_imgs
        print(
            'Avg. MSE: {} | Avg. MAE: {}'.format
            (
                mean_mse, mean_mae 
            )
        )

    # Convert lists to tensors for evaluation
    i_features = torch.stack(all_features['i_f'])
    t_features = torch.stack(all_features['t_f'])
    i_uncertainty = torch.stack(all_features['i_u']).mean(dim=1)
    t_uncertainty = torch.stack(all_features['t_u']).mean(dim=1)
    all_classes = np.array(all_classes)

    # Compute recall scores
    pred_ranks = get_pred_ranks(i_features, t_features, recall_ks=recall_ks)
    recall_scores = get_recall(pred_ranks, recall_ks=recall_ks, n_gallery_per_query=1)
    print(f"Recall scores: {recall_scores}")

    # Sort by uncertainty
    sort_v_idx, sort_t_idx = sort_wrt_uncer({'i_u': i_uncertainty, 't_u': t_uncertainty})

    # Create uncertainty bins
    i_bins = create_uncer_bins_eq_samples(sort_v_idx, n_bins=n_bins)
    t_bins = create_uncer_bins_eq_samples(sort_t_idx, n_bins=n_bins)

    print("Uncertainty bin analysis completed.")
    return {
        'mean_mse': mean_mse,
        'mean_mae': mean_mae,
        'recall_scores': recall_scores,
        'i_bins': i_bins,
        't_bins': t_bins,
    }


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
    print(f"Loading checkpoint from {ckpt_path}...")
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


def load_and_evaluate_with_uncertainty(
    ckpt_path='../ckpt/ProbVLM_Net_best.pth',
    dataset="coco",
    data_dir="../datasets/coco",
    batch_size=64,
    device='cuda',
    n_bins=10,
    recall_ks=(1, 5, 10),
):
    # Load data loaders
    dataloader_config = mch({
        "batch_size": batch_size,
        "random_erasing_prob": 0,
        "traindata_shuffle": True
    })

    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    eval_loader = loaders['val']

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
    print(f"Loading checkpoint from {ckpt_path}...")
    ProbVLM_Net.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Evaluate with the new function
    metrics = eval_ProbVLM_with_uncertainty(
        CLIP_Net,
        ProbVLM_Net,
        eval_loader,
        device=device,
        n_bins=n_bins,
        recall_ks=recall_ks,
    )

    # Print the results
    print(f"Evaluation Metrics: {metrics}")

    return metrics


def main():
    load_and_evaluate()


if __name__ == "__main__":
    main()

