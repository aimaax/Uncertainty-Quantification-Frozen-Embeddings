import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from networks_BBB_EncBL import *

def parse_captions(file_path):
    """
    Parses captions_val2024.json to extract image and caption IDs.
    
    Args:
        file_path (str): Path to the captions_val2024.json file.
    
    Returns:
        all_image_ids (list): List of image IDs.
        all_caption_ids (list): List of caption IDs.
    """
    all_image_ids = []
    all_caption_ids = []
    
    with open(file_path, 'r') as f:
        data = f.read()
        if data.startswith(','):
            data = data[1:]
        json_data = json.loads(f"[{data}]")

    for item in json_data:
        all_image_ids.append(item["image_id"])
        all_caption_ids.append(item["id"])
    
    return all_image_ids, all_caption_ids


def uncert_est(
    ckpt_path="../ckpt/ProbVLM_Net_best.pth",
    dataset="coco",
    data_dir="../datasets/coco",
    model_type="ProbVLM",
    batch_size=64,
    n_fw=10,
    device="cuda"
):
    dataloader_config = mch({
        "batch_size": batch_size,
        "random_erasing_prob": 0,
        "traindata_shuffle": True
    })
    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    valid_loader = loaders['val']

    # Load CLIP and BayesCap networks
    CLIP_Net = load_model(device=device, model_path=None)
    if model_type == "BBB":
        Net = BayesCap_for_CLIP(inp_dim=512, out_dim=512, hid_dim=256, num_layers=3)
    elif model_type == "ProbVLM":
        Net = BayesCap_for_CLIP_ProbVLM(inp_dim=512, out_dim=512, hid_dim=256, num_layers=3)
    else:
        raise ValueError(f"Unknown model type '{model_type}'")

    print(f"Loading checkpoint from {ckpt_path}")
    Net.load_state_dict(torch.load(ckpt_path, map_location=device))

    CLIP_Net.to(device)
    CLIP_Net.eval()
    Net.to(device)
    Net.eval()

    # Collect embeddings and uncertainties
    img_embs, cap_embs = [], []
    img_sigmas, cap_sigmas = [], []

    print("Extracting features and uncertainties...")
    with torch.no_grad():
        for (xI, xT) in tqdm(valid_loader, unit="batch"):
            xI, xT = xI.to(device), xT.to(device)
            xfI, xfT = CLIP_Net(xI, xT)
            (img_mu, img_alpha, img_beta), (txt_mu, txt_alpha, txt_beta) = Net(xfI, xfT)

            img_embs.append(img_mu.cpu().numpy())
            cap_embs.append(txt_mu.cpu().numpy())
            img_sigmas.append(img_alpha.cpu().numpy())  # Assuming alpha represents uncertainty
            cap_sigmas.append(txt_alpha.cpu().numpy())  # Same assumption

    # Stack into arrays
    img_embs = np.vstack(img_embs)
    cap_embs = np.vstack(cap_embs)
    img_sigmas = np.vstack(img_sigmas)
    cap_sigmas = np.vstack(cap_sigmas)

    # Compute similarities
    print("Computing similarity matrix...")
    sims = compute_matmul_sims(img_embs, cap_embs)

    # Parse captions and test IDs
    captions_path = os.path.join(data_dir, "annotations", "captions_val2024.json")
    test_ids_path = os.path.join(data_dir, "annotations", "coco_test_ids.npy")
    all_iids, all_cids = parse_captions(captions_path)
    print("all iids: ",all_iids)
    print("all cids: ", all_cids)
    test_ids = np.load(test_ids_path)

    # Filter IDs
    filtered_indices = [i for i, img_id in enumerate(all_iids) if img_id in test_ids]
    filtered_iids = np.array([all_iids[i] for i in filtered_indices])
    filtered_cids = np.array([all_cids[i] for i in filtered_indices])

    # Evaluate uncertainties
    print("Evaluating uncertainties...")
    report_dict, unc_vs_scores = eval_coco_uncertainty(
        sims, img_sigmas, cap_sigmas, data_dir, n_bins=10
    )

    for key, value in report_dict.items():
        print(f"{key}: {value}")

    return report_dict


def main():
    model = "../ckpt/ProbVLM_Net_best.pth"
    dataset = "coco"
    data_dir = "../datasets/coco"
    report = uncert_est(ckpt_path=model, dataset=dataset, data_dir=data_dir, model_type="ProbVLM")
    print("Uncertainty evaluation completed.")
    print(report)

