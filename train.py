import torch
from src.engine.device import get_device
from torch.utils.data import DataLoader
from src.data.dataset import BDD_Dataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.engine.criterion import get_criterion, get_optimizer
from src.models.model import get_model
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.utils.config import get_config
import os
import yaml

# config
ROOT_DIR = "data/processed"
BATCH_SIZE = 8
EPOCHS = 10

# def train(train_img_dir, train_label_dir, val_img_dir, val_label_dir, checkpoint_dir, output_dir, batch_size=8, epochs=10):
def train(cfg):
    # use gpu
    DEVICE = get_device()

    now = datetime.now()
    format = "%m-%d-%Y-%H%M"
    now_str = now.strftime(format)

    # directory to store model checkpoints
    output_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(output_dir, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")

    batch_size = cfg["training"]["batch_size"]
    true_batch_size = cfg["training"]["true_batch_size"]

    # mIoU metric to display during training
    train_miou_metric = MulticlassJaccardIndex(num_classes=cfg["project"]["num_classes"], ignore_index=cfg["project"]["ignore_index"]).to(DEVICE)
    val_miou_metric = MulticlassJaccardIndex(num_classes=cfg["project"]["num_classes"], ignore_index=cfg["project"]["ignore_index"]).to(DEVICE)

    # load data
    train_ds = BDD_Dataset(cfg["data"]["train"]["img"], cfg["data"]["train"]["label"], transform=get_train_transforms(cfg))
    val_ds = BDD_Dataset(cfg["data"]["val"]["img"], cfg["data"]["val"]["label"], transform=get_val_transforms(cfg))

    train_loader = DataLoader(train_ds, batch_size=true_batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=true_batch_size, shuffle=False, num_workers=1)

    # setup model
    model = get_model(num_classes=cfg["project"]["num_classes"]).to(DEVICE)

    # setup loss function and optimizer
    criterion = get_criterion()

    # load optimizer
    current_lr = cfg["training"]["lr"]
    print("current_lr 1st:", current_lr)
    optimizer = get_optimizer(model, lr=current_lr, weight_decay=cfg["training"]["weight_decay"])

    # drop learning rate by 10x if validation loss doesn't improve for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg["training"]["scheduler_factor"], patience=cfg["training"]["scheduler_patience"]
    )

    # get most recent training run
    checkpoint_loaded = False
    existing_runs = sorted([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])

    if len(existing_runs) > 0:
        latest_train_run = os.path.join(output_dir, existing_runs[-1])
        latest_checkpoint = os.path.join(latest_train_run, "best_model.pth")
    
        # load the checkpoint if it exists
        if os.path.exists(latest_checkpoint):
            checkpoint_loaded = True
            print(f"Loading existing checkpoint {latest_checkpoint}...")
            checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Loaded optimizer state: learning rate={current_lr}")

            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("Loaded scheduler state.")

            # start at epoch where left off
            start_epoch = checkpoint.get("epoch", 0) + 1

            # recrod best val loss for saving checkpoints
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"Baseline val loss: {best_val_loss:.4f}")   
        else:
            print("No previous checkpoint found.") 
    
    # store results from this training run
    new_checkpoint_dir = os.path.join(output_dir, f"train_run_{now_str}") # format: train_run_01-01-2026-0000
    os.makedirs(new_checkpoint_dir, exist_ok=True)

    # if checkpoint was loaded, load the training log and append to it
    if checkpoint_loaded:
        print("Loading existing training logs...")
        old_log_path = os.path.join(latest_train_run, "training_log.csv")
        history = pd.read_csv(old_log_path).to_dict('records')
    else: # else start a new training log
        history = []

    # early stopping
    early_stop_patience = cfg["training"]["patience"] # use config
    early_stop_counter = 0

    accumulation_steps = batch_size // true_batch_size
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        # training
        model.train() # set model to training mode
        train_loss = 0
   
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        iter = 0
        for i, (images, masks, _) in enumerate(train_pbar):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # forward pass
            outputs = model(images)
            raw_loss = criterion(outputs, masks)
            loss = raw_loss / accumulation_steps
            
            # calculate mIoU
            preds = torch.argmax(outputs, dim=1)
            train_miou_metric.update(preds, masks)

            # backward pass
            loss.backward()
            
            # gradient accumulation
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # update loss
            train_loss += raw_loss
            
            # update progress bar every 10 batches
            if i % 10 == 0:
                train_pbar.set_postfix({"loss": f"{raw_loss.item():.4f}", "lr": f"{current_lr:.1e}"})

            iter += 1

        # validation
        avg_train_loss = (train_loss / len(train_loader)).item()
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
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(new_checkpoint_dir, "best_model.pth"))

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
        log_path = os.path.join(new_checkpoint_dir, "training_log.csv")
        df = pd.DataFrame(history)
        df.to_csv(log_path, index=False)
        
        # plot loss and mIoU
        plot_history(df, new_checkpoint_dir)

        # check if should stop early
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Training halted.")
            break

    # save a copy of the config used for this training run
    config_save_path = os.path.join(new_checkpoint_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"Configuration saved to {config_save_path}")

# plot loss and mIoU
def plot_history(df, checkpoint_dir):
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
    # plt.savefig("checkpoints/learning_curves.png")
    plt.savefig(os.path.join(checkpoint_dir, "learning_curves.png"))
    plt.close() # close to save memory

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
    cfg = get_config()
    train(cfg)