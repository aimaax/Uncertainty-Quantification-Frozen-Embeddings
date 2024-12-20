"""
https://www.pinecone.io/learn/series/image-search/zero-shot-image-classification-clip/
https://arxiv.org/abs/2103.00020

Don't forget to cite

"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))
from tqdm.auto import tqdm
from utils_asym import *
import numpy as np
import torch
from clip import *
from ds.vocab import Vocabulary
from model import AsymProbAdaptor
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.stats as stats
from model import  AsymProbAdaptor
# Add the path two levels up to sys.path
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, parent_folder)

from edward.models import AsymProbAdaptorBNN

def multi_fwpass_ProbVLM(
    BayesCap_Net,
    xfI, xfT,
    n_fw=15,
    device="cuda"
):
    txt_mu_lst, txt_variance_lst, img_mu_lst = [], [], []
    BayesCap_Net.eval()
    BayesCap_Net.to(device)

    for i in range(n_fw):
        #(txt_mu, txt_log_variance, img_mu) = BayesCap_Net(xfI, xfT)
        txt_mu, txt_log_variance, img_mu = BayesCap_Net(xfT.to(device), xfI.to(device))
        txt_mu_lst.append(txt_mu.unsqueeze(0))
        txt_variance_lst.append(torch.exp(txt_log_variance.unsqueeze(0)))
        img_mu_lst.append(img_mu.unsqueeze(0))
    ##
    img_mu_lst = torch.cat(img_mu_lst, dim=0)
    img_mu_var = torch.var(img_mu_lst, dim=0) # epistemic

    img_mu_mean = torch.mean(img_mu_lst, dim=0)

    txt_variance_lst = torch.cat(txt_variance_lst, dim=0)
    txt_variance_mu = torch.mean(txt_variance_lst, dim=0) # aleatoric

    txt_mu_mean = torch.mean(torch.cat(txt_mu_lst, dim=0), dim=0)
    return (img_mu_var, img_mu_mean, txt_variance_mu, txt_mu_mean)

def get_features_uncer_ProbVLM_asym(
    CLIP_Net,
    BayesCap_Net,
    t_loader,
    device='cuda'
):
    r_dict= {
        'i_clip_f': [],
        't_clip_f': [],
        'i_adapt_f':[],
        't_adapt_f':[],
        'e_u': [],
        'a_u': [],
    }
    # extract all features
    with torch.no_grad():
        for i_inputs, t_inputs, class_labels, _ in tqdm(t_loader):
            n_batch = i_inputs.shape[0]
            i_inputs, t_inputs = i_inputs.to(device), t_inputs.to(device)
            img_em = CLIP_Net.encode_image(i_inputs)
            text_em  = CLIP_Net.encode_text(t_inputs)
            outs = multi_fwpass_ProbVLM(BayesCap_Net, img_em, text_em)
            # (img_mu_var, img_mu_mean, txt_variance_mu, txt_mu_mean)
            for j in range(n_batch):
                r_dict['i_clip_f'].append(img_em[j,:])
                r_dict['t_clip_f'].append(text_em[j,:])
                r_dict['i_adapt_f'].append(outs[1][j,:])
                r_dict['t_adapt_f'].append(outs[3][j,:])
                r_dict['e_u'].append(outs[0][j,:])
                r_dict['a_u'].append(outs[2][j,:])
    return r_dict

def image_retrieval_eval(CLIP_Net, Net, device="cuda"):
    # Perform accuracy rejection guided by aleatoric uncertainty for dataset coco and flickr
    dataloader_config = mch({
        "batch_size": 64,
        "random_erasing_prob": 0,
        "traindata_shuffle": True
    })

    # load coco
    dataset = "coco"
    data_dir = "../datasets/coco"
    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    data_valid_loader = loaders['val']
    rejection_accuracy_coco, rejection_percentage = reject_accuracy(data_valid_loader, CLIP_Net, Net)

    # load flickr
    dataset = "flickr"
    data_dir = "../../../datasets/flickr"
    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    data_valid_loader = loaders['val']
    rejection_accuracy_flickr, _ = reject_accuracy(data_valid_loader, CLIP_Net, Net)

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(rejection_percentage, rejection_accuracy_coco, marker='o', label = "coco")
    plt.plot(rejection_percentage, rejection_accuracy_flickr, marker='o', label = "flickr")
    plt.legend()
    plt.xlabel('Percentage of Captions Rejected')
    plt.ylabel('Retrieval Accuracy')
    plt.grid(True)
    plt.savefig("img_retrieval_BNN.png")


def reject_accuracy(coco_valid_loader, CLIP_Net, Net, device="cuda"):

    true_preds = []
    preds = []  # This will store predicted indices
    predicted_classes = []
    predicted_label_embs = []
    aleatoric_uncertainties = []
    epistemic_uncertainties = []
    img_emb_list = []

    r_dict = get_features_uncer_ProbVLM_asym(CLIP_Net, Net, coco_valid_loader)
    all_image = torch.stack(r_dict['i_adapt_f'])  # Shape: (num_samples, embedding_dim)
    al_list = []

    for i_f, t_f, e_u, a_u in zip(r_dict['i_adapt_f'], r_dict['t_adapt_f'], r_dict['e_u'], r_dict['a_u']):
        image = i_f
        caption = t_f

        # Cosine similarity, text to image
        scores = torch.matmul(caption, all_image.T)
        pred_img_idx = torch.argmax(scores).item() # Max one is the retreival

        true_preds.append(1 if torch.equal(image, all_image[pred_img_idx]) else 0)

        preds.append(pred_img_idx)

        # Save mean of al uncertainty
        al_list.append(torch.mean(a_u).item())

    rejection_percentages = np.arange(0, 96, 5)
    rejection_accuracies = []
    remaining_samples_sizes = []
    true_positive_count = []

    # Sort indices based on aleatoric uncertainty
    sorted_indices = np.argsort(al_list)[::-1]

    for rejection_pct in rejection_percentages:
        num_samples_to_reject = int(len(al_list) * rejection_pct / 100)
        num_samples_to_reject = min(num_samples_to_reject, len(al_list) - 1)

        remaining_indices = sorted_indices[num_samples_to_reject:]
        remaining_size = len(remaining_indices)

        remaining_true_preds = np.array(true_preds)[remaining_indices]
        remaining_preds = np.array(preds)[remaining_indices]

        # Compute true positives and rejection accuracy
        true_positive = np.sum(remaining_true_preds == 1)
        true_positive_count.append(true_positive)

        rejection_accuracy = np.sum(remaining_true_preds == 1) / len(remaining_true_preds)
        rejection_accuracies.append(rejection_accuracy)
        remaining_samples_sizes.append(remaining_size)

        print(f"Rejection {rejection_pct} %: \t Accuracy: {rejection_accuracy}")

    rho, p_value =  stats.spearmanr(rejection_percentages, rejection_accuracies)
    print("Spearman corrleation", rho)

    return rejection_accuracies, rejection_percentages


def main():
    CLIP_Net, _ = clip.load("ViT-B/32", device='cuda')
    model = "BNN"
    device="cuda"
    if model =="DO":
        ckpt_path = "../../../../edward/best_model_100_epochs_flickr.pth"
        Net = AsymProbAdaptor().to(device).half()
        Net.load_state_dict(torch.load(ckpt_path, map_location=device))
        Net.to(device)
    elif model == "BNN":
        ckpt_path = "../../../../edward/best_model_BNN_COCO_two_BL.pth"
        Net = AsymProbAdaptorBNN().to(device).float()
        Net.load_state_dict(torch.load(ckpt_path, map_location=device))
        Net.to(device)
        CLIP_Net = CLIP_Net.float()
    print(f"Loaded checkpoint from {ckpt_path}")

    image_retrieval_eval(CLIP_Net, Net)


if __name__ == "__main__":
    main()

