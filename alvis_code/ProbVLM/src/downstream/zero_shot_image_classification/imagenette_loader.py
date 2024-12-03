import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path


class NoisyImagenette:
    def __init__(self, data_dir, csv_path, batch_size=32, image_size=224):
        """
        Initialize the NoisyImagenette dataset class for zero-shot classification.
        """
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / csv_path
        self.batch_size = batch_size
        self.image_size = image_size

        # Define the label map
        self.label_map = {
            'n02979186': 'cassette_player',
            'n03417042': 'garbage_truck',
            'n01440764': 'tench',
            'n02102040': 'English_springer',
            'n03028079': 'church',
            'n03888257': 'parachute',
            'n03394916': 'French_horn',
            'n03000684': 'chain_saw',
            'n03445777': 'golf_ball',
            'n03425413': 'gas_pump'
        }

        # Load and process annotations
        self.df_annot = self._load_annotations()

        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_annotations(self):
        """
        Load and preprocess annotations from the CSV file.
        """
        df_annot = pd.read_csv(self.csv_path)
        df_annot = df_annot[['path', 'noisy_labels_0']].rename({'noisy_labels_0': 'label', 'path': 'img_filename'},
                                                               axis='columns')
        df_annot['label'] = df_annot['label'].map(self.label_map)
        return df_annot

    def get_loader(self):
        """
        Get a DataLoader object for the entire dataset.
        """

        class NoisyImagenetteDataset(Dataset):
            def __init__(self, annotations, data_dir, transform=None):
                self.annotations = annotations
                self.data_dir = data_dir
                self.transform = transform

            def __len__(self):
                return len(self.annotations)

            def __getitem__(self, idx):
                row = self.annotations.iloc[idx]
                img_path = self.data_dir / row['img_filename']
                label = row['label']
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label

        dataset = NoisyImagenetteDataset(self.df_annot, self.data_dir, self.transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return loader

