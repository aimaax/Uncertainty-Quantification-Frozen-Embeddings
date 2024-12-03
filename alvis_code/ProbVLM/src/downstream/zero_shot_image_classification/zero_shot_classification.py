import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))
from tqdm.auto import tqdm
from utils import *
import numpy as np
import torch
from clip import *
from networks_BBB_EncBL import BayesCap_for_CLIP_BBB_Enc
from imagenette_loader import *
from transformers import CLIPProcessor  # Add this import
import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader

def load_CLIP():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_Net = load_model(device='cuda', model_path=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_Net.to(device)

    if isinstance(CLIP_Net, torch.nn.DataParallel):
        CLIP_Net = CLIP_Net.module

    return processor, CLIP_Net

def load_adapter(model_name="BBB_Enc", device="cuda"):
    Net = BayesCap_for_CLIP_BBB_Enc(
        inp_dim=512,
        out_dim=512,
        hid_dim=256,
        num_layers=3
    ).to(device)

    return Net

def load_imagenette_loader(data_dir, csv_path, batch_size=32, image_size=224):
    data_handler = NoisyImagenette(
        data_dir=data_dir,
        csv_path=csv_path,
        batch_size=batch_size,
        image_size=image_size
    )

    data_loader = data_handler.get_loader()

    return data_loader, data_handler


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


def prepare_imagenette_labels(data_handler, CLIP_Net, device="cuda"):
    labels = list(data_handler.label_map.values())
    clip_labels = [f"a photo of a {label}" for label in labels]

    label_tokens = tokenize(clip_labels, context_length=77).to(device)

    label_emb = CLIP_Net.encode_text(label_tokens)
    label_emb = label_emb.detach().cpu().numpy()
    label_emb = label_emb / np.linalg.norm(label_emb, axis=0)

    return labels, clip_labels, label_emb


def prepare_cifar100_labels(dataset, CLIP_Net, device="cuda"):
    labels = dataset.classes  # CIFAR-100 class names
    clip_labels = [f"a photo of a {label}" for label in labels]

    label_tokens = tokenize(clip_labels, context_length=77).to(device)

    label_emb = CLIP_Net.encode_text(label_tokens)
    label_emb = label_emb.detach().cpu().numpy()
    label_emb = label_emb / np.linalg.norm(label_emb, axis=0)

    return labels, clip_labels, label_emb


def zero_shot_classifier_imagenette(data_loader, label_emb, CLIP_Net, Net=None, device="cuda"):
    true_preds = []
    preds = []

    for images, labels in tqdm(data_loader):
        images = images.to(device)

        img_emb, _= CLIP_Net.encode_image(images, is_weights=True)
        
        if Net != None:
            img_emb, _ = Net(img_emb, t_features=None)
            img_emb = img_emb[0] # extract mu

        img_emb = img_emb.detach().cpu().numpy()

        # similarity scores
        scores = np.dot(img_emb, label_emb.T)
        pred_classes = np.argmax(scores, axis=1)
        preds.extend(pred_classes)

        for i in range(len(labels)):
            true_label = labels[i]  
            predicted_class_idx = pred_classes[i]  
            predicted_class = labels[predicted_class_idx]  
            true_preds.append(1 if true_label == predicted_class else 0)

    accuracy = calc_accuracy(true_preds)
    return accuracy


def zero_shot_classifier_cifar_100(data_loader, label_emb, global_labels, CLIP_Net, Net=None, device="cuda"):
    true_preds = []
    preds = []

    for images, batch_labels in tqdm(data_loader):
        images = images.to(device)

        # Encode image embeddings
        img_emb, _ = CLIP_Net.encode_image(images, is_weights=True)

        if Net is not None:
            img_emb, _ = Net(img_emb, t_features=None)
            img_emb = img_emb[0]  # Extract mu

        img_emb = img_emb.detach().cpu().numpy()

        # Compute similarity scores
        scores = np.dot(img_emb, label_emb.T)
        pred_classes = np.argmax(scores, axis=1)
        preds.extend(pred_classes)

        # Compare predictions with true labels
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


def main():
    processor, CLIP_Net = load_CLIP()
    Net = load_adapter()

    # imagenette2
    """
    data_loader, data_handler = load_imagenette_loader(data_dir="../../../datasets/imagenette2-320", csv_path="noisy_imagenette.csv")
    labels, clip_labels, label_emb = prepare_imagenette_labels(data_handler, CLIP_Net)
    accuracy = zero_shot_classifier(data_loader, label_emb, CLIP_Net, Net=None)
    """

    # cifar-100
    train_loader, train_dataset = load_cifar100_loader(batch_size=32, image_size=224, train=True)
    test_loader, test_dataset = load_cifar100_loader(batch_size=32, image_size=224, train=False)
    labels, clip_labels, label_emb = prepare_cifar100_labels(test_dataset, CLIP_Net)

    accuracy = zero_shot_classifier_cifar_100(test_loader, label_emb, labels, CLIP_Net, Net=None)

    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()


