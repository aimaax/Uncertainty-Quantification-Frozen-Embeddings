from torchvision.datasets import VisionDataset
import os
from PIL import Image
import json


class Flickr30k(VisionDataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = os.path.expanduser(root)
        self.ann_file = os.path.expanduser(ann_file)

        # Read annotations from file
        self.annotations = self._load_data()

        self.ids = list(sorted(self.annotations.keys()))
    
    def _load_data(self):
        with open(self.ann_file, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            data[k] = v['comments']
        return data
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Load image
        filename = os.path.join(self.root, img_id)
        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        caption = self.annotations[img_id]
        if self.target_transform is not None:
            caption = self.target_transform(caption)

        return image, caption

