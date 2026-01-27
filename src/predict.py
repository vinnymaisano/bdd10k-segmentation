import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.data.transforms import get_val_transforms
from src.utils.color_map import get_bdd_palette
# from models.model import get_model

# setup
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# MODEL_PATH = "checkpoints/best_model.pth"
# IMAGE_PATH = "data/processed/images/val/9b970e47-ce2164fd.jpg"

# color map
COLOR_MAP = get_bdd_palette()

def predict(model, image_path, device, color_map=COLOR_MAP):
    # load model
    # model = get_model(num_classes=19).to(DEVICE)
    # model.load_state_dict(torch.load(model_path, map_location=DEVICE)["model_state_dict"])
    model.eval()

    # load and transform image
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2] # 720, 1280
    
    # resize to 256, 512 and normalize
    transform = get_val_transforms()
    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        output = model(input_tensor)
        # tensor shape: (batch_size, channels, height, width)
        # argmax(1) (over channels) picks the class ID (0-18) with the highest probability for each pixel
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # upscale back to 1280 x 720
    full_res_mask = cv2.resize(
        mask.astype(np.uint8),
        (w_orig, h_orig),
        interpolation=cv2.INTER_NEAREST
    )

    # map IDs to colors
    color_mask = COLOR_MAP[full_res_mask].astype(np.uint8)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

    # overlay on original image for "X-Ray" effect
    overlay = cv2.addWeighted(image_bgr, 0.2, color_mask, 0.8, 0)

    return overlay