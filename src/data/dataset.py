import os
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

class BDDDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
            root_dir (str/Path): Path to 'data/processed'
            split (str): 'train' or 'val'
            transform: Albumentations transformation pipeline
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images" / split
        self.mask_dir = self.root_dir / "labels" / split
        self.transform = transform
        
        # create list of all .jpg files in the image directory
        if not self.img_dir.exists():
            raise RuntimeError(f"Directory not found: {self.img_dir}")
            
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # get the filename for the image
        img_name = self.images[idx]
        img_path = self.img_dir / img_name
        
        # create the mask filename by swapping extensions and adding the suffix
        # example: 'abc.jpg' -> 'abc_train_id.png'
        mask_name = img_name.replace(".jpg", "_train_id.png")
        mask_path = self.mask_dir / mask_name

        # load the image (.jpg), convert to RGB
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load the mask (.png)
        # load as IMREAD_GRAYSCALE to get a single channel of class IDs (0-18)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # force mask to be the same shape as the image
        if image.shape[:2] != mask.shape[:2]:
            # must use inter_nearest to prevent class labels from being smoothed over
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) 
        
        if mask is None:
            raise FileNotFoundError(f"Could not find mask for {img_name} at {mask_path}")

        # apply Albumentations transforms
        # handles resizing and normalizing both image and mask together
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        img_name = self.images[idx]
        # convert to PyTorch tensors
        # images are typically floats, masks must be longs (integers) for the loss function
        return image, mask.to(dtype=torch.long), img_name