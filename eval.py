import os
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
from src.data.dataset import BDDDataset
from src.data.transforms import get_val_transforms
from src.engine.criterion import get_criterion
from src.engine.device import get_device
from datetime import datetime

def evaluate():
    device = get_device()
    print(f"Using device: {device}")

    now = datetime.now()
    format = "%m-%d-%Y-%H%M"
    now_str = now.strftime(format)

    # latest model checkpoint
    latest_train_run = sorted(os.listdir("checkpoints"))[-1]
    checkpoint_path = os.path.join("checkpoints", latest_train_run, "best_model.pth")

    # dataset to evaluate model on
    split = "val"

    # load model
    model = get_model(num_classes=19).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.eval()

    # load data
    batch_size = 4
    transform = get_val_transforms()
    
    val_ds = BDDDataset('data/processed', split=split, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size)


    # directories for outputting metrics and masks
    output_dir = f'eval/{split}_evaluation_results_{now_str}' # save metrics
    mask_dir = f'preds/{split}_masks_{now_str}' # save predictions
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

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

            preds_np = preds.cpu().numpy().astype(np.uint8)
            # save each prediction
            for j in range(preds_np.shape[0]):
                save_path = os.path.join(mask_dir, filenames[j].replace(".jpg", ".png"))
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
    evaluate()