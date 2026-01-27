import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=(448, 768)):
    """
    Enhanced augmentations to close the generalization gap on BDD10K.
    """
    return A.Compose([
        # 1. SPATIAL VARIETY
        A.RandomResizedCrop(
            size=(img_size[0], img_size[1]),
            scale=(0.5, 1.0),
            p=1.0
        ),
        
        # 2. GEOMETRIC ROBUSTNESS
        # Replaced ShiftScaleRotate with Affine to resolve UserWarning.
        # scale_limit=0.2 becomes scale=(0.8, 1.2)
        # shift_limit=0.0625 becomes translate_percent=(-0.0625, 0.0625)
        A.Affine(
            translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
            scale=(0.8, 1.2),
            rotate=(-10, 10),
            p=0.5,
            border_mode=0 # Equivalent to border_mode=0 (constant padding)
        ),
        
        A.HorizontalFlip(p=0.5),

        # 3. LIGHTING & WEATHER ROBUSTNESS
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.CLAHE(clip_limit=2, p=1.0),
        ], p=0.4),
        
        # 4. SENSOR NOISE
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(p=1.0),
        ], p=0.2),

        # 5. FINAL STEPS
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transforms(img_size=(448, 768)):
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])