import cv2
from src.utils.color_map import get_bdd_palette

def colorize_mask(mask_array):
    """
    Converts a grayscale mask (H, W) with class IDs 
    to an RGB image (H, W, 3) using the palette.
    """
    palette = get_bdd_palette()
    color_mask = palette[mask_array] 
    return color_mask

def create_overlay(original_img, color_mask, alpha=0.5):
    """Blends the mask over the original image."""
    # Ensure both are same size and type
    overlay = cv2.addWeighted(original_img, 1 - alpha, color_mask, alpha, 0)
    return overlay