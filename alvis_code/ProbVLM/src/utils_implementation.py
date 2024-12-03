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
import matplotlib.pyplot as plt

def compute_average_uncertainty_per_feature(uncertainty_values):
    """
    Compute the average epistemic uncertainty per feature and print summary statistics.
    
    Args:
        uncertainty_values (torch.Tensor): Tensor of shape (n_fw, batch_size, num_features), 
                                            where `n_fw` is the number of forward passes.
    
    Returns:
        torch.Tensor: Average uncertainty per feature, shape (num_features,).
    """
    # Step 1: Average over forward passes
    uncertainty_avg_over_fw = torch.mean(uncertainty_values, dim=0)  # Shape: (batch_size, num_features)
    
    # Step 2: Average over batches
    uncertainty_avg_per_feature = torch.mean(uncertainty_avg_over_fw, dim=0)  # Shape: (num_features,)

    # Compute variance for additional statistics
    uncertainty_variance_per_feature = torch.var(uncertainty_avg_over_fw, dim=0)  # Shape: (num_features,)

    # Print summary statistics
    print("Uncertainty Per Feature - Mean:", torch.mean(uncertainty_avg_per_feature).item())
    print("Uncertainty Per Feature - Min:", torch.min(uncertainty_avg_per_feature).item())
    print("Uncertainty Per Feature - Max:", torch.max(uncertainty_avg_per_feature).item())
    print("Uncertainty Per Feature - Variance (Overall):", torch.mean(uncertainty_variance_per_feature).item())
    
    return uncertainty_avg_per_feature

def uncertainty_multiFWP(Net, CLIP_out_i, CLIP_out_t, n_fw=15, model="ProbVLM", print_stuff=False):
    i_mu_vals, i_alpha_vals, i_beta_vals = [], [], []
    t_mu_vals, t_alpha_vals, t_beta_vals = [], [], []

    i_aleatoric, t_aleatoric = [], []

    Net.eval()
    if model == "ProbVLM":
        for layer in Net.children():
            for l in layer.modules():
                if(isinstance(l, nn.Dropout)):
                    l.p = 0.3
                    l.train()

    for i in range(n_fw):
        # sample from the probabilistic network
        (i_mu, i_alpha, i_beta), (t_mu, t_alpha, t_beta) = Net(CLIP_out_i, CLIP_out_t)

        i_mu_vals.append(i_mu.unsqueeze(0))
        i_alpha_vals.append(i_alpha.unsqueeze(0))
        i_beta_vals.append(i_beta.unsqueeze(0))
        i_aleatoric.append(aleatoric_uncertainty(i_alpha, i_beta).unsqueeze(0))

        t_mu_vals.append(t_mu.unsqueeze(0))
        t_alpha_vals.append(t_alpha.unsqueeze(0))
        t_beta_vals.append(t_beta.unsqueeze(0))
        t_aleatoric.append(aleatoric_uncertainty(t_alpha, t_beta).unsqueeze(0))

    # concatenate
    i_mu_cat = torch.cat(i_mu_vals, dim=0)
    i_alpha_cat = torch.cat(i_alpha_vals, dim=0)
    i_beta_cat = torch.cat(i_beta_vals, dim=0)
    i_aleatoric_cat = torch.cat(i_aleatoric, dim=0)

    t_mu_cat = torch.cat(t_mu_vals, dim=0)
    t_alpha_cat = torch.cat(t_alpha_vals, dim=0)
    t_beta_cat = torch.cat(t_beta_vals, dim=0)
    t_aleatoric_cat = torch.cat(t_aleatoric, dim=0)

    # calculate mean and variance
    i_mu_mean, i_mu_var = torch.mean(i_mu_cat, dim=0), torch.var(i_mu_cat, dim=0)
    i_alpha_mean, i_alpha_var = torch.mean(i_alpha_cat, dim=0), torch.var(i_alpha_cat, dim=0)
    i_beta_mean, i_beta_var = torch.mean(i_beta_cat, dim=0), torch.var(i_beta_cat, dim=0)
    i_aleatoric_mean = torch.mean(i_aleatoric_cat, dim=0)

    t_mu_mean, t_mu_var = torch.mean(t_mu_cat, dim=0), torch.var(t_mu_cat, dim=0)
    t_alpha_mean, t_alpha_var = torch.mean(t_alpha_cat, dim=0), torch.var(t_alpha_cat, dim=0)
    t_beta_mean, t_beta_var = torch.mean(t_beta_cat, dim=0), torch.var(t_beta_cat, dim=0)
    t_aleatoric_mean = torch.mean(t_aleatoric_cat, dim=0)

    if print_stuff:
        i_avg_uncertainty_per_feature = compute_average_uncertainty_per_feature(i_mu_cat)
        print("Average uncertainty per feature:", i_avg_uncertainty_per_feature)

    return (i_mu_mean, i_mu_var, i_aleatoric_mean), (t_mu_mean, t_mu_var, t_aleatoric_mean)
    
    
def aleatoric_uncertainty(alpha, beta):
    return (1/alpha)**2 * torch.exp(torch.lgamma(3/beta + 1e-3)) / torch.exp(torch.lgamma(1/beta + 1e-3))


def get_uncertainties(CLIP_Net, Net, t_loader, n_fw=15, model="ProbVLM", device="cuda"):
    """
    Size of i_mu at forward pass 50: torch.Size([64, 512])
    Size of i_mu_cat after concatenation: torch.Size([50, 64, 512])
    Size of i_mu_mean: torch.Size([64, 512])
    Size of i_mu_var: torch.Size([64, 512])
    """

    uncert_dict = {
            "ir_f" : [],
            "i_eu" : [],
            "i_au" : [],
            "tr_f" : [],
            "t_eu" : [],
            "t_au" : []
    }

    with torch.no_grad():
        #counter = 0
        for i_inputs, t_inputs, class_labels, _ in tqdm(t_loader):
            """
            counter += 1
            if counter == 20:
                return uncert_dict
            """
            #print(counter)
            # prepare CLIP
            n_batch = i_inputs.shape[0]
            i_inputs, t_inputs = i_inputs.to(device), t_inputs.to(device)
            outputs = CLIP_Net(i_inputs, t_inputs)

            # get uncertainties
            outs = uncertainty_multiFWP(Net, outputs[0], outputs[1], n_fw=n_fw, print_stuff=False)

            for j in range(n_batch):
                uncert_dict['ir_f'].append(outs[0][0][j,:]) # query/gallery
                uncert_dict['i_eu'].append(outs[0][1][j,:])
                #uncert_dict['i_au'].append()
                uncert_dict['tr_f'].append(outs[1][0][j,:]) # query/gallery
                uncert_dict['t_eu'].append(outs[1][1][j,:])
                #uncert_dict['t_au'].append()

    return uncert_dict


def summarize_uncertainties(uncert_dict):
    summary = {}
    for key, values in uncert_dict.items():
        if key != "i_au" and key != "t_au":
            values_np = torch.cat(values).cpu().numpy() 
            summary[key] = {
                "mean": np.mean(values_np),
                "std": np.std(values_np),
                "min": np.min(values_np),
                "max": np.max(values_np),
            }

    return summary


def get_pred_ranks(q_features, g_features, recall_ks=(1,5,10)):
    """
    Args:
        q_features (torch.tensor, size=[#query, embedding dim])
        g_features (torch.tensor, size=[#gallery, embedding dim])
        recall_ks (list[:int] or tuple[:int])
    Returns:
        pred_ranks_all (np.ndarray, size=[#query, max(recall_ks)]):
            data indices of similarity ranking in descending order
    """
    max_k = max(recall_ks)
    n_q_features = len(q_features)

    pred_ranks_all = []
    for idx in range(n_q_features):
        sims = (q_features[idx : idx + 1] @ g_features.t())
        _, pred_ranks = torch.topk(sims, k=max_k, dim=-1)
        pred_ranks_all.append(pred_ranks)
    pred_ranks_all = torch.cat(pred_ranks_all, dim=0).cpu().numpy()

    return pred_ranks_all


def get_recall(pred_ranks_all, recall_ks=(1,5,10), n_gallery_per_query=5):
    """
    Args:
        pred_ranks_all (np.ndarray, size=[#query, max(recall_ks)]): 
            data indices of similarity ranking in descending order
        recall_ks (list[:int] or tuple[:int])
        n_gallery_per_query (float)
    Returns:
        recall_scores (list[:float]): list of recall@k
    """
    existence = lambda arr1, arr2: any([i in arr2 for i in arr1])
    def gt_idxs(query_idx):
        if n_gallery_per_query >= 1:
            return np.arange(query_idx * n_gallery_per_query, 
                             (query_idx + 1) * n_gallery_per_query)
        else:
            return np.array([int(query_idx * n_gallery_per_query)])

    recall_scores = []
    for recall_k in recall_ks:
        score = sum([existence(pred_ranks[:recall_k], gt_idxs(query_idx))
                     for query_idx, pred_ranks in enumerate(pred_ranks_all)]) / len(pred_ranks_all)
        recall_scores.append(score)

    return recall_scores


def get_recall_COCOFLICKR(pred_ranks_all, recall_ks=(1,5,10), n_gallery_per_query=5, q_idx=None):
    """
    Args:
        pred_ranks_all (np.ndarray, size=[#query, max(recall_ks)]): 
            data indices of similarity ranking in descending order
        recall_ks (list[:int] or tuple[:int])
        n_gallery_per_query (float)
    Returns:
        recall_scores (list[:float]): list of recall@k
    """
    existence = lambda arr1, arr2: any([i in arr2 for i in arr1])
    def gt_idxs(query_idx):
        if n_gallery_per_query >= 1:
            return np.arange(query_idx * n_gallery_per_query, 
                             (query_idx + 1) * n_gallery_per_query)
        else:
            return np.array([int(query_idx * n_gallery_per_query)])

    recall_scores = []
    for recall_k in recall_ks:
        score = sum([existence(pred_ranks[:recall_k], q_idx)
                     for query_idx, pred_ranks in enumerate(pred_ranks_all)]) / len(pred_ranks_all)
        recall_scores.append(score)

    return recall_scores


def new_recall(pred_ranks_all,recall_ks=(1,5,10),q_classes_all=None,g_classes_all=None):
    recall_scores = []
    for recall_k in recall_ks:
        corr=0
        total = len(pred_ranks_all)
        for i in range(len(pred_ranks_all)):
            gt_class = q_classes_all[i]
            pred_classes = [g_classes_all[j] for j in pred_ranks_all[i][:recall_k]]
            if gt_class in pred_classes:
                corr+=1
        recall_scores.append(corr/total)

    return recall_scores


def sort_wrt_uncer(r_dict):
    orig_v_idx = {}
    for i in range(len(r_dict['i_eu'])):
        orig_v_idx[i] = torch.mean(r_dict['i_eu'][i]).item()
    sort_v_idx = sorted(orig_v_idx.items(), key=lambda x: x[1], reverse=True)
    
    orig_t_idx = {}
    for i in range(len(r_dict['t_eu'])):
        orig_t_idx[i] = torch.mean(r_dict['t_eu'][i]).item()
    sort_t_idx = sorted(orig_t_idx.items(), key=lambda x: x[1], reverse=True)
    
    return sort_v_idx, sort_t_idx

def create_uncer_bins_eq_spacing(sort_idx, n_bins=10):
    max_uncer = sort_idx[0][1]
    min_uncer = sort_idx[-1][1]
    
    step_uncer = np.linspace(min_uncer, max_uncer, num=n_bins)
    print('uncer_steps: ', step_uncer)
    
    ret_bins = {'bin{}'.format(i):[] for i in range(n_bins)}
    
    for val in sort_idx:
        idx, uv = val
        for j, step in enumerate(step_uncer):
            if uv<=step:
                ret_bins['bin{}'.format(j)].append(val)
    return ret_bins

def create_uncer_bins_eq_samples(sort_idx, n_bins=10):
    sort_idx = sort_idx[::-1]
    ret_bins = {'bin{}'.format(i):[] for i in range(n_bins)}
    n_len = len(sort_idx)
    z = 0
    for i, val in enumerate(sort_idx):
        if i<=z+(n_len//n_bins):
            ret_bins['bin{}'.format(int(z//(n_len/n_bins)))].append(val)
        else:
            z += n_len//n_bins
    return ret_bins

def compute_recall_for_uncertainty_bins(uncert_dict, recall_k=1, n_bins=5):
    sort_v_idx, sort_t_idx = sort_wrt_uncer(uncert_dict)

    # Create bins based on uncertainty levels
    bins = create_uncer_bins_eq_samples(sort_t_idx, n_bins=n_bins)
    #bins = create_uncer_bins_eq_spacing(sort_t_idx, n_bins=n_bins)

    recall_values = []

    # Iterate through bins and calculate recall@1 for each bin
    for bin_key, samples in bins.items():
        if not samples:
            recall_values.append(0)  # If bin is empty, append 0
            continue
        # Extract indices for the current bin
        indices = [sample[0] for sample in samples]  # Extract indices from sorted list
        bin_query_features = torch.stack([uncert_dict['tr_f'][i] for i in indices])  # Query features for the bin
        bin_gallery_features = torch.stack(uncert_dict['ir_f'])  # All gallery features

        # Get the ranking of predicted results for these query features
        pred_ranks_all = get_pred_ranks(bin_query_features, bin_gallery_features, recall_ks=(recall_k,))

        # Compute the recall@1 for this bin
        recall_scores = get_recall(pred_ranks_all, recall_ks=(recall_k,))
        recall_values.append(recall_scores[0])  # Recall@1 is the first value

    return recall_values, bins

def plot_recall_vs_uncertainty(recall_values, n_bins=5):
    # Plot Recall@1 against Epistemic Uncertainty
    bin_centers = np.arange(n_bins)  # Bin indices
    print("Recall values: ", recall_values)
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, recall_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Epistemic Uncertainty Bins')
    plt.ylabel('Recall@1')
    plt.title('Recall@1 vs Epistemic Uncertainty')
    plt.grid(True)
    plt.xticks(bin_centers)
    plt.savefig("../figs/test.png")

