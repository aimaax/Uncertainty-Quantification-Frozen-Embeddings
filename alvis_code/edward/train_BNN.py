import torch
from tqdm import tqdm 
import clip
from torchvision.datasets.coco import CocoCaptions
import torchbnn as bnn
from models import AsymProbAdaptorBNN
from utils import collate_fn
from torch.utils.data import DataLoader
from dataset import Flickr30k
from validate import validate

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()
adaptor = AsymProbAdaptorBNN().to(device).float()

dataset = CocoCaptions(
     root='/mimer/NOBACKUP/Datasets/Microsoft-COCO/train2017',
     annFile='/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/captions_train2017.json',
     transform=preprocess)

dataset_validate = CocoCaptions(
     root='/mimer/NOBACKUP/Datasets/Microsoft-COCO/val2017',
     annFile='/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/captions_val2017.json',
     transform=preprocess)

"""
dataset = Flickr30k(
    root='../ProbVLM/datasets/flickr/flickr30k_images',
    ann_file='../annotations.json', 
    transform=preprocess)
"""

batch_size = 64
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True,
    num_workers=4, pin_memory=True, prefetch_factor=2)

dataloader_validate = DataLoader(
    dataset_validate, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True,
    num_workers=4, pin_memory=True, prefetch_factor=2)

num_epochs = 100
patience = 20
patience_counter = 0

metrics = {
    "train_mae": [],
    "train_mse": [],
    "val_mae": [],
    "val_mse": []
}

parameters = {
    "txt_mu": [],
    "txt_log_variance": [],
    "img_mu": []
}

train_mae, train_mse, val_mae, val_mse = 0, 0, 0, 0
txt_mu, txt_log_variance, img_mu = 0, 0, 0

optimizer = torch.optim.SGD(adaptor.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, num_epochs*len(dataloader), 1e-5)

steps = 0
min_val_loss = float('inf')  
model_name = "BNN_COCO"
best_model_path = "best_model_" + model_name + ".pth" 
last_model_path = "last_model_" + model_name + ".pth" 
metrics_path = model_name + "_metrics.npy"
parameters_path = model_name + "_parameters.npy"

adaptor.train()
kl_loss = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    train_mse_list = []
    train_mae_list = []

    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
        for images, captions in dataloader:
            adaptor.train()

            images = images.to(device).float()
            captions = clip.tokenize(captions, context_length=77, truncate=True).to(device)

            image_features = model.encode_image(images).float()
            text_features = model.encode_text(captions).float()

            txt_mu, txt_log_var, img_mu = adaptor(text_features, image_features)

            loss, kl_loss = adaptor.loss(txt_mu, txt_log_var, img_mu, image_features)
            epoch_loss += loss.item()

            mse = ((img_mu - image_features) ** 2).mean().item()
            mae = torch.abs(img_mu - image_features).mean().item()

            train_mse_list.append(mse)
            train_mae_list.append(mae)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            tepoch.set_postfix(loss=loss.item())
            tepoch.update(1)  

            steps += 1

    avg_train_mse = sum(train_mse_list) / len(train_mse_list)
    avg_train_mae = sum(train_mae_list) / len(train_mae_list)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss:.4f}, Training MSE: {avg_train_mse:.4f}, Training MAE: {avg_train_mae:.4f}")
    print(f"KL Loss: {kl_loss}")
    num_params = sum(p.numel() for p in adaptor.parameters())
    print(num_params)

    torch.save(adaptor.state_dict(), last_model_path)

    val_mse, val_mae = validate(model, adaptor, dataloader_validate, device)
    print(f"Validation Results - Epoch {epoch + 1}: MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")

    metrics["train_mse"].append(avg_train_mse)
    metrics["train_mae"].append(avg_train_mae)
    metrics["val_mse"].append(val_mse)
    metrics["val_mae"].append(val_mae)
    parameters["txt_mu"].append(txt_mu)
    parameters["txt_log_variance"].append(txt_log_variance)
    parameters["img_mu"].append(img_mu)

    if val_mse < min_val_loss:
        min_val_loss = val_mse
        patience_counter = 0
        torch.save(adaptor.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

np.save(metrics_path, metrics)
print(f"Metrics saved to {metrics_path}.")
np.save(parameters_path, parameters)
print(f"Parameters saved to {parameters_path}.")


