import yaml
import os

def is_kaggle():
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None

def get_config():
    # load base yaml
    filename = "config.yaml"
    with open(filename, "r") as f:
        cfg = yaml.safe_load(f)

    # resolve environment
    env = "kaggle" if is_kaggle() else "local"
    root = cfg["data"]["roots"][env]
    
    # resolve data paths
    for split in ["train", "val"]:
        cfg["data"][split]["img"] = os.path.join(root, cfg["data"][split]["img"])
        cfg["data"][split]["label"] = os.path.join(root, cfg["data"][split]["label"]) 
        
        # validation
        img_path = cfg["data"][split]["img"]
        label_path = cfg["data"][split]["label"]
        
        assert os.path.isdir(img_path), f"Image directory missing: {img_path}"
        assert os.path.isdir(label_path), f"Label directory missing: {label_path}"

    # create checkpoint directory on kaggle
    if env == "kaggle":
        cfg["training"]["checkpoint_dir"] = "/kaggle/working/checkpoints"

    print(f"Using {env.upper()} environment. Data root: {root}")
    return cfg