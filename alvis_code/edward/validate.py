import numpy as np
from clip import *
import torch

def validate(model, adaptor, dataloader, device='cuda'):
    model.eval()
    adaptor.eval()

    mse_list = []
    mae_list = []

    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            captions = clip.tokenize(captions, context_length=77, truncate=True).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(captions)

            txt_mu, txt_log_var, img_mu = adaptor(text_features, image_features)

            mse = ((img_mu - image_features) ** 2).mean().item()
            mae = torch.abs(img_mu - image_features).mean().item()

            mse_list.append(mse)
            mae_list.append(mae)

        mse = np.mean(mse_list)
        mae = np.mean(mae_list)

        return mse, mae

