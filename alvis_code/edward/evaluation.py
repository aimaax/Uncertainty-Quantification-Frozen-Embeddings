import sys
import os
from tqdm.auto import tqdm
import numpy as np
import torch
from clip import *
from model import AsymProbAdaptor 
import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from torchvision.datasets.coco import CocoCaptions
from utils import collate_fn
from torch.utils.data import DataLoader

def load_image(image_path):
    return Image.open(image_path)

def load_adaptor(ckpt_path, device="cuda"):
    adaptor = AsymProbAdaptor().to(device).half()
    adaptor.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {ckpt_path}")

    return adaptor

def load_cifar100_loader(batch_size=32, image_size=224, train=True):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),  # CIFAR-100 normalization
    ])

    dataset = CIFAR100(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader, dataset

def prepare_cifar100_labels(dataset, CLIP_Net, device="cuda"):
    labels = dataset.classes  # CIFAR-100 class names
    clip_labels = [f"a photo of a {label}" for label in labels]

    label_tokens = clip.tokenize(clip_labels, context_length=77).to(device)

    label_emb = CLIP_Net.encode_text(label_tokens)
    label_emb = label_emb.detach().cpu().numpy()
    label_emb = label_emb / np.linalg.norm(label_emb, axis=0)

    return labels, clip_labels, label_emb

def zero_shot_classifier_cifar_100(data_loader, label_emb, global_labels, CLIP_Net, Net=None, device="cuda"):
    true_preds = []
    preds = []

    for images, batch_labels in tqdm(data_loader):
        images = images.to(device)

        img_emb = CLIP_Net.encode_image(images)
        if Net is not None:
            outs = Net(None, img_emb)
            img_emb = outs[2] 

        img_emb = img_emb.detach().cpu().numpy()

        # cosine similarity
        scores = np.dot(img_emb, label_emb.T)
        pred_classes = np.argmax(scores, axis=1)
        preds.extend(pred_classes)

        for i in range(len(batch_labels)):
            true_label = batch_labels[i]  # Ground-truth index
            predicted_class_idx = pred_classes[i]  # Predicted global index
            predicted_class = global_labels[predicted_class_idx]  # Correctly reference global labels
            true_preds.append(1 if true_label == predicted_class_idx else 0)

    accuracy = calc_accuracy(true_preds)
    return accuracy

def calc_accuracy(true_preds):
    accuracy = sum(true_preds) / len(true_preds)

    return accuracy

def multi_fwpass(
    BayesCap_Net,
    xfI, n_fw=15,
    return_mu_lst=False
):
    img_mu_lst =  []

    for i in range(n_fw):
        (txt_mu, txt_log_variance, img_mu) = BayesCap_Net(None, xfI)
        img_mu_lst.append(img_mu.unsqueeze(0))

    img_mu_lst_cat = torch.cat(img_mu_lst, dim=0)
    img_mu_var = torch.var(img_mu_lst_cat, dim=0) 

    if return_mu_lst:
        return img_mu_var, img_mu_lst
    else:
        return img_mu_var

def feature_uncer(
    BayesCap_Net,
    img_emb_list,
    device='cuda',
    n_fw=15
):
    e_u = []

    with torch.no_grad():
        for img_emb in tqdm(img_emb_list):
            if isinstance(img_emb, np.ndarray):
                img_emb = torch.tensor(img_emb, dtype=torch.float32)
            img_emb = img_emb.to(device)  
            img_var = multi_fwpass(BayesCap_Net, img_emb.unsqueeze(0), n_fw=n_fw)  
            e_u.append(img_var.mean().item())

    return e_u

def reject_accuracy(data_loader, label_emb, clip_labels, global_labels, CLIP_Net, Net=None, device="cuda"):
    """
    label_emb: after clip and normalized
    clip_labels: that is passed into clip
    global_labels: original cifar labels
    """
    true_preds = []
    preds = []
    predicted_classes = []
    predicted_label_embs = []  
    aleatoric_uncertainties = []  
    epistemic_uncertainties = []
    img_emb_list = []

    Net.eval()

    counter = 0 
    for images, batch_labels in tqdm(data_loader):
        images = images.to(device)

        img_emb = CLIP_Net.encode_image(images)

        outs = Net(None, img_emb)
        img_emb = outs[2] 

        img_emb = img_emb.detach().cpu().numpy()

        # cosine similarity
        scores = np.dot(img_emb, label_emb.T)
        pred_classes = np.argmax(scores, axis=1)
        preds.extend(pred_classes)

        for i in range(len(batch_labels)):
            true_label = batch_labels[i]  # ground-truth index
            predicted_class_idx = pred_classes[i]  # predicted global index
            predicted_class = global_labels[predicted_class_idx]  # correctly reference global labels
            true_preds.append(1 if true_label == predicted_class_idx else 0)

            predicted_label_emb = label_emb[predicted_class_idx]
            predicted_label_embs.append(predicted_label_emb)

        for emb in img_emb:
            img_emb_list.append(emb)

    for predicted_label_emb in tqdm(predicted_label_embs):
        predicted_label_emb_tensor = torch.tensor(predicted_label_emb).to(device)

        outs = Net(predicted_label_emb_tensor.unsqueeze(0), None)
        log_var = outs[1].detach().cpu().numpy()
        log_var = np.mean(np.exp(log_var))
        aleatoric_uncertainties.append(log_var) 

    epistemic_uncertainties = feature_uncer(Net, img_emb_list, device=device, n_fw=30)

    total_uncertainties = []
    for epistemic, aleatoric in zip(epistemic_uncertainties, aleatoric_uncertainties):
        #total_uncertainty = epistemic + aleatoric
        #total_uncertainty = epistemic
        #total_uncertainty = aleatoric
        total_uncertainty = (aleatoric * epistemic) ** (1/2)
        total_uncertainties.append(total_uncertainty)

    rejection_percentages = np.arange(0, 96, 5)  
    rejection_accuracies = []
    remaining_samples_sizes = []
    true_positive_count = []

    sorted_indices = np.argsort(total_uncertainties)[::-1]  

    for rejection_pct in rejection_percentages:
        num_samples_to_reject = int(len(total_uncertainties) * rejection_pct / 100)
        num_samples_to_reject = min(num_samples_to_reject, len(total_uncertainties) - 1)

        remaining_indices = sorted_indices[num_samples_to_reject:]
        remaining_size = len(remaining_indices)

        remaining_true_preds = np.array(true_preds)[remaining_indices]
        remaining_preds = np.array(preds)[remaining_indices]

        true_positive = np.sum(remaining_true_preds == 1)
        true_positive_count.append(true_positive)

        rejection_accuracy = np.sum(remaining_true_preds==1) / len(remaining_true_preds)
        rejection_accuracies.append(rejection_accuracy)
        remaining_samples_sizes.append(remaining_size)

        print(f"Rejection {rejection_pct} %: \t Accuracy: {rejection_accuracy}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(rejection_percentages, rejection_accuracies, marker='o')
    plt.title('Rejection Accuracy vs. Aleatoric Uncertainty')
    plt.xlabel('Percentage of Samples Rejected')
    plt.ylabel('Accuracy After Rejection')
    plt.grid(True)
    plt.savefig("accuracy_rejection.png")


def experiments(Net, CLIP_Net, processor, device='cuda'):
    """
    manual_labels = [
        "a photo",
        "a photo of a cat",
        "a photo of a black cat on a table",
        "a photo of a black cat on a table with a glass of pepsi beside it",
        "a photo of a black cat on a table with a glass of pepsi beside it, and a man in the backgroud",
    ]
    """
    manual_labels = [
        "an image",
        "an image of a man",
        "an image of a man drinking",
        "an image of a man drinking a bottle",
        "an image of a man drinking a bottle of water",
        "an image of a man drinking a bottle of water while sitting on a bench",
        "an image of a man drinking a bottle of water while sitting on a brown bench",
    ]
    
    label_tokens = clip.tokenize(manual_labels, context_length=77).to(device)
    label_emb = CLIP_Net.encode_text(label_tokens)

    aleatoric_uncertainties = []
    
    for emb in tqdm(label_emb):
        label_emb_tensor = torch.tensor(emb).to(device)

        with torch.no_grad():
            Net.eval()
            outs = Net(label_emb_tensor.unsqueeze(0), None)

        log_var = outs[1].detach().cpu().numpy()  
        log_var = np.mean(np.exp(log_var))  
        aleatoric_uncertainties.append(log_var)

    for label, uncertainty in zip(manual_labels, aleatoric_uncertainties):
        print(f"Aleatoric uncertainty for '{label}': {uncertainty}")

    return aleatoric_uncertainties

def calculate_epistemic_uncertainty_for_images(image1_path, image2_path, Net, CLIP_Net, processor, device='cuda', n_fw=15):
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")
    
    image1_processed = processor(image1).unsqueeze(0).to(device)  
    image2_processed = processor(image2).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        Net.eval()
        for layer in Net.children():
            for l in layer.modules():
                if isinstance(l, nn.Dropout):
                    l.p = 0.7
                    l.train()

        img_emb1 = CLIP_Net.encode_image(image1_processed)  
        img_emb2 = CLIP_Net.encode_image(image2_processed)  
    
    uncertainties_image1, mu_lst1 = multi_fwpass(Net, img_emb1, return_mu_lst=True, n_fw=5000)  
    uncertainties_image2, mu_lst2 = multi_fwpass(Net, img_emb2, return_mu_lst=True, n_fw=5000)  

    mu_lst1 = torch.stack(mu_lst1).squeeze(dim=1).squeeze(dim=1).cpu().detach().numpy()
    mu_lst2 = torch.stack(mu_lst2).squeeze(dim=1).squeeze(dim=1).cpu().detach().numpy()
    uncertainties_image1 = uncertainties_image1.cpu().detach().numpy()
    uncertainties_image2 = uncertainties_image2.cpu().detach().numpy()

    mu_lst1 = mu_lst1.mean(axis=1)
    mu_lst2 = mu_lst2.mean(axis=1)

    uncertainties_image1_mean = uncertainties_image1.mean(axis=1)
    uncertainties_image2_mean = uncertainties_image2.mean(axis=1)

    print(f"Mean Epistemic Uncertainty for Image 1: {uncertainties_image1_mean.mean()}")
    print(f"Mean Epistemic Uncertainty for Image 2: {uncertainties_image2_mean.mean()}")

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(mu_lst1, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Epistemic Uncertainty Distribution for Image 1")
    plt.xlabel('mu')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(mu_lst2, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title("Epistemic Uncertainty Distribution for Image 2")
    plt.xlabel('mu')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig("epistemic_uncertainty_distributions.png")
    print("Saved figure to 'epistemic_uncertainty_distributions.png'")

def main():
    model, preprocess = clip.load("ViT-B/32", device='cuda')
    ckpt_path = "best_model_1.pth"
    Net = load_adaptor(ckpt_path)

    #experiments(Net, CLIP_Net, processor)
    image1 = 'FLICKR_EXAMPLE.jpg'
    image2 = 'interstellar.jpg'

    calculate_epistemic_uncertainty_for_images(image1, image2, Net, model, preprocess, device="cuda", n_fw=15)
    #experiments(Net, model, preprocess)

    #train_loader, train_dataset = load_cifar100_loader(batch_size=32, image_size=224, train=True)
    #test_loader, test_dataset = load_cifar100_loader(batch_size=32, image_size=224, train=False)
    #labels, clip_labels, label_emb = prepare_cifar100_labels(test_dataset, model)

    #accuracy = zero_shot_classifier_cifar_100(test_loader, label_emb, labels, model, Net=Net)
    #print(f"Accuracy: {accuracy}")

    #rejection_accuracies = reject_accuracy(test_loader, label_emb, clip_labels, labels, model, Net=Net)

if __name__ == "__main__":
    main()



