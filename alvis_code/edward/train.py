import torch
from tqdm import tqdm 
import clip
from torchvision.datasets.coco import CocoCaptions
from model import AsymProbAdaptor
from utils import collate_fn
from torch.utils.data import DataLoader
from dataset import Flickr30k


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
adaptor = AsymProbAdaptor().to(device).half()

"""
dataset = CocoCaptions(
     root='/mimer/NOBACKUP/Datasets/Microsoft-COCO/train2017',
     annFile='/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/captions_train2017.json',
     transform=preprocess)
"""

dataset = Flickr30k(
    root='../ProbVLM/datasets/flickr/flickr30k_images',
    ann_file='../annotations.json', 
    transform=preprocess)

batch_size = 64
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True,
    num_workers=4, pin_memory=True, prefetch_factor=2)

num_epochs = 100

optimizer = torch.optim.SGD(adaptor.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, num_epochs*len(dataloader), 1e-5)

steps = 0
best_loss = float('inf')  
best_model_path = "best_model_100_epochs_flickr.pth"  
last_model_path = "last_model_100_epochs_flickr.pth" 

for epoch in range(num_epochs):
    epoch_loss = 0
    counter = 0
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
        for images, captions in dataloader:
            adaptor.train()

            images = images.to(device)
            captions = clip.tokenize(captions, context_length=77, truncate=True).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(captions)

            txt_mu, txt_log_var, img_mu = adaptor(text_features, image_features)

            loss = adaptor.loss(txt_mu, txt_log_var, img_mu, image_features)
            epoch_loss += loss.item()
            counter += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            tepoch.set_postfix(loss=loss.item())
            tepoch.update(1)  

            """
            if steps % 200 == 0:
                with torch.no_grad():
                    adaptor.eval()

                    sentences = [
                        "a photo",
                        "a photo of a cat",
                        "a photo of a black cat",
                        "a photo of a black cat on a table",
                        "a photo of a black cat on a table with a glass of pespi beside it",
                        "a photo of a black cat on a table with a glass of pespi beside it, and a man in the background",
                    ]
                    text = clip.tokenize(sentences).to(device)
                    txt_mu, txt_log_var, img_mu = adaptor(model.encode_text(text), None)
                    print(f'Step {steps}:')
                    for j in range(len(sentences)):
                        print(f'{sentences[j]}: {txt_log_var[j].exp().mean().item()}')
                    print("===================================", flush=True)
            """

            steps += 1

    average_epoch_loss = epoch_loss / counter
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_epoch_loss:.4f}")

    torch.save(adaptor.state_dict(), last_model_path)

    if average_epoch_loss < best_loss:
        best_loss = average_epoch_loss
        torch.save(adaptor.state_dict(), best_model_path)
        print(f"Best model updated and saved at {best_model_path}")

