import torch

# customise the collate function for the dataloader
def collate_fn(batch):
    raw_images, caption_lists = zip(*batch)
    captions = []
    images = []
    for idx, caption_list in enumerate(caption_lists):
        # filter out samples with less than 5 captions
        if len(caption_list) < 5:
            # print(f"Sample {idx} has less than 5 captions, skipping")
            continue
        captions.extend(caption_list[:5])
        images.append(raw_images[idx])
    images = torch.stack(images, dim=0)
    return images, captions
