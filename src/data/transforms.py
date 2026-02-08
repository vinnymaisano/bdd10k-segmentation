import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(cfg):
    trans_cfg = cfg["transforms"]
    aug = trans_cfg["augmentations"]
    """
    Enhanced augmentations to close the generalization gap on BDD10K.
    """
    return A.Compose([
        # spatial robustness
        A.RandomResizedCrop(
            size=(trans_cfg["resize_h"], trans_cfg["resize_w"]),
            scale=aug["crop_scale"],
            p=1.0
        ),

        # horizontal flip
        A.HorizontalFlip(p=aug["horizontal_flip_prob"]),
        
        # geometric robustness
        A.Affine(
            translate_percent={"x": (-aug["affine_translate_pct"], aug["affine_translate_pct"]),
                               "y": (-aug["affine_translate_pct"], aug["affine_translate_pct"])},
            scale=aug["affine_scale_range"],
            rotate=aug["affine_rotate_deg"],
            p=aug["affine_prob"],
            border_mode=0, # constant padding
            fill=cfg["project"]["ignore_index"] # ignore_index
        ),

        # lighting and weather robustness
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"], saturation=aug["saturation"], hue=aug["hue"], p=1.0),
            A.RandomShadow(p=aug["shadow_prob"]),
            A.CLAHE(clip_limit=aug["clahe_limit"], p=1.0),
        ], p=aug["weather_prob"]),
        
        # sensor noise
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(p=1.0),
        ], p=aug["noise_prob"]),

        # normalize
        A.Normalize(
            mean=trans_cfg["mean"],
            std=trans_cfg["std"]
        ),
        ToTensorV2()
    ])

def get_val_transforms(cfg):
    trans_cfg = cfg["transforms"]
    
    return A.Compose([
        A.Resize(height=trans_cfg["resize_h"], width=trans_cfg["resize_w"]),
        A.Normalize(
            mean=trans_cfg["mean"],
            std=trans_cfg["std"]
        ),
        ToTensorV2()
    ])