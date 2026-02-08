import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import json
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from torchmetrics.classification import MulticlassConfusionMatrix
from src.models.model import get_model 
from src.data.dataset import BDD_Dataset
from src.data.transforms import get_val_transforms
from src.engine.criterion import get_criterion
from src.engine.device import get_device
from datetime import datetime
from src.utils.config import get_config

def evaluate(checkpoint_dir, img_dir, mask_dir, output_dir, save_preds):
    device = get_device()
    print(f"Using device: {device}")

    # timestamp to uniquely identify this evaluation
    now = datetime.now()
    format = "%m-%d-%Y-%H%M"
    now_str = now.strftime(format)

    # if no checkpoint directory provided, default to the latest one in checkpoints folder
    if checkpoint_dir is None:
        if len(os.listdir("checkpoints")) == 0:
            raise FileNotFoundError(
                f"No checkpoint_dir provided, so defaulted to /checkpoints folder but folder is empty.",
                "Please provide a path via --checkpoint_dir or ensure /checkpoints folder isn't empty"
            )
        latest_train_run = sorted(os.listdir("checkpoints"))[-1]
        checkpoint_dir = os.path.join("checkpoints", latest_train_run, "best_model.pth")

    # load model
    model = get_model(num_classes=19).to(device)
    model.load_state_dict(torch.load(checkpoint_dir, map_location=device)["model_state_dict"])
    model.eval()

    # load data
    batch_size = 4
    cfg = get_config
    transform = get_val_transforms(cfg)
    
    val_ds = BDD_Dataset(img_dir, mask_dir, transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # get dataset from image path
    split = ""
    if "val" in str(img_dir):
        split = "val_"
    elif "train" in str(img_dir):
        split = "train_"
    elif "test" in str(img_dir):
        split = "test_"
    print(f"Dataset split: {split}")

    # folder for outputs
    output_dir = os.path.join(output_dir, f"{split}evaluation_results_{now_str}")
    os.makedirs(output_dir, exist_ok=True)
    print("output folder:", output_dir) 

    # folder for predicted masks
    preds_dir = os.path.join(output_dir, "preds")
    if save_preds:
        os.makedirs(preds_dir, exist_ok=True)

    classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    # metrics
    conf_matrix_metric = MulticlassConfusionMatrix(num_classes=19).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=19, average='none').to(device)
    acc_metric = MulticlassAccuracy(num_classes=19, average='micro').to(device)

    # criterion
    criterion = get_criterion()
    total_loss = 0.0

    # run evaluation and save predictions
    with torch.no_grad():
        for images, masks, filenames in tqdm(val_loader, desc="Evaluating", unit="batch"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            # calculate loss
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            iou_metric.update(preds, masks)
            acc_metric.update(preds, masks)
            conf_matrix_metric.update(preds, masks)
            
            if save_preds:
                preds_np = preds.cpu().numpy().astype(np.uint8)
                # save each prediction
                for j in range(preds_np.shape[0]):
                    save_path = os.path.join(preds_dir, filenames[j].replace(".jpg", ".png"))
                    cv2.imwrite(save_path, preds_np[j])
            
    print(f"Saved {len(os.listdir(mask_dir))} raw masks to {mask_dir}")

    # compute metrics
    cm = conf_matrix_metric.compute().cpu().numpy()
    per_class_iou = iou_metric.compute().cpu().numpy()
    mIoU = per_class_iou.mean()
    avg_loss = total_loss / len(val_loader)
    total_acc = acc_metric.compute().item()

    summary = {
        "avg_loss": float(avg_loss),
        "global_pixel_accuracy": float(total_acc),
        "mIoU": float(mIoU)
    }

    # export
    with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)

    results_dict = {
        "class_name": classes,
        "iou": per_class_iou
    }
    df = pd.DataFrame(results_dict)
    csv_path = os.path.join(output_dir, "per_class_iou.csv")
    df.to_csv(csv_path, index=False)

    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)

    print("\n--- Final evaluation report ---")
    for name, score in zip(classes, per_class_iou):
        print(f"{name:15}: {score.item():.4f}")
    print(f"\nOVERALL mIoU: {per_class_iou.mean().item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=False, 
        help="Path to the model checkpoint that will be used to make predicitons. Default is latest checkpoint"
    )
    parser.add_argument(
        "--img_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing input images"
    )
    parser.add_argument(
        "--mask_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing ground truth masks"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to directory that metrics and predictions will be saved to"
    )
    parser.add_argument(
        "--save_preds",
        action="store_true",
        help="Save the output masks"
    )
    args = parser.parse_args()

    checkpoint_dir = None
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir

    print(args)

    evaluate(checkpoint_dir=checkpoint_dir, img_dir=Path(args.img_dir), mask_dir=Path(args.mask_dir), output_dir=Path(args.output_dir), save_preds=args.save_preds)