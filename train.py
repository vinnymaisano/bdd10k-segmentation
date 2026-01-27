import torch
from torch.utils.data import DataLoader
from src.data.dataset import BDDDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.engine.criterion import get_criterion, get_optimizer
from src.models.model import get_model
import torch.nn as nn
from tqdm import tqdm
# from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import MulticlassJaccardIndex
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# config
ROOT_DIR = "data/processed"
BATCH_SIZE = 8
EPOCHS = 10
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train():
    now = datetime.now()
    format = "%m-%d-%Y-%H%M"
    now_str = now.strftime(format)

    start_epoch = 0
    best_val_loss = float("inf")

    # store model checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    # mIoU metric to display during training
    train_miou_metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255).to(DEVICE)
    val_miou_metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255).to(DEVICE)

    # load data
    train_ds = BDDDataset(ROOT_DIR, split="train", transform=get_train_transforms())
    val_ds = BDDDataset(ROOT_DIR, split="val", transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # setup model
    model = get_model(num_classes=19).to(DEVICE)

    # setup loss function and optimizer
    criterion = get_criterion()

    # load optimizer
    current_lr = 1e-4
    optimizer = get_optimizer(model, lr=current_lr, weight_decay=1e-5)

    latest_train_run = f"checkpoints/{sorted(os.listdir("checkpoints"))[-1]}"
    latest_checkpoint_path = os.path.join(latest_train_run, "best_model.pth")
    
    # load checkpoint if it exists
    if os.path.exists(latest_checkpoint_path):
        print("Loading existing checkpoint...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            current_lr = optimizer.param_groups[0]['lr']

        start_epoch = checkpoint.get("epoch", 0)

        # for saving checkpoints
        best_val_loss, _ = validate(model, val_loader, criterion, DEVICE, val_miou_metric)
        print(f"Baseline val loss: {best_val_loss:.4f}")

    # store results from this training run 
    new_train_run_dir = f"checkpoints/train_run_{now_str}"
    os.makedirs(new_train_run_dir, exist_ok=True)

    # drop learning rate by 10x if validation loss doesn't improve for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    # storage for logs
    log_path = os.path.join(new_train_run_dir, "training_log.csv")
    print(log_path)
    if os.path.exists(log_path):
        print("Loading existing training logs...")
        history = pd.read_csv(log_path).to_dict('records')
    else:
        history = []

    # early stopping
    early_stop_patience = 7
    early_stop_counter = 0

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        # training
        model.train() # set model to training mode
        train_loss = 0
   
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        iter = 0
        for images, masks, _ in train_pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # calculate mIoU
            preds = torch.argmax(outputs, dim=1)
            train_miou_metric.update(preds, masks)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # update loss
            train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item(), "lr": f"{current_lr:.1e}"})

            iter += 1

        # validation
        avg_train_loss = train_loss / len(train_loader)
        final_train_miou = train_miou_metric.compute().item()
        avg_val_loss, final_val_miou = validate(model, val_loader, criterion, DEVICE, val_miou_metric, epoch)

        scheduler.step(avg_val_loss)

        # get current LR for logs
        current_lr = optimizer.param_groups[0]["lr"]
        
        # display epoch summary
        print(f"Epoch {epoch+1} summary:")
        print(f"Training | Loss: {avg_train_loss:.4f} | mIoU: {final_train_miou:.4f}")
        print(f"Validation | Loss: {avg_val_loss:.4f} | mIoU: {final_val_miou:.4f}")

        # reset for next epoch
        train_miou_metric.reset()
        val_miou_metric.reset()

        # save weights if this is the best model so far
        if avg_val_loss < best_val_loss:
            early_stop_counter = 0 # reset counter
            best_val_loss = avg_val_loss

            # save model and optimizer state
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'epoch': epoch
            }
            torch.save(checkpoint, "checkpoints/best_model.pth")

            print(f"--> New best model saved: (val loss: {best_val_loss:.4f} | val mIoU: {final_val_miou})")
        else:
            early_stop_counter += 1
            print(f"--> No improvement for {early_stop_counter} epochs.")
        
        # append to hisotry
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_miou": final_train_miou,
            "val_miou": final_val_miou,
            "lr": current_lr
        }
        history.append(epoch_metrics)

        # save to csv
        df = pd.DataFrame(history)
        df.to_csv("checkpoints/training_log.csv", index=False)
        
        # plot loss and mIoU
        plot_history(df)

        # check if should stop early
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Training halted.")
            break

        # clear GPU cache
        # if DEVICE.type == "mps":
        #     torch.mps.empty_cache()

# plot loss and mIoU
def plot_history(df):
    # update plot
    plt.figure(figsize=(12, 5))

    # loss plot
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()

    # mIoU Plot
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_miou'], label='Train mIoU')
    plt.plot(df['epoch'], df['val_miou'], label='Val mIoU')
    plt.title('mIoU Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig("checkpoints/learning_curves.png")
    plt.close() # Close to save memory

def validate(model, loader, criterion, device, miou_metric, epoch=None):
    model.eval()
    val_loss = 0
    miou_metric.reset() # Ensure metric is clean

    with torch.no_grad():
        desc = "Validating" if not epoch else f"Epoch {epoch+1} [Val]"
        for images, masks, _ in tqdm(loader, desc=desc, leave=False):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            miou_metric.update(preds, masks)

    avg_loss = val_loss / len(loader)
    miou = miou_metric.compute().item()
    
    return avg_loss, miou

if __name__ == "__main__":
    train()