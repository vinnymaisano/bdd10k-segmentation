import yaml
import os

def is_kaggle():
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None

def get_config():
    # load base yaml
    filename = "config.yaml"
    with open(filename, "r") as f:
        cfg = yaml.safe_load(f)

    # resolve paths based on the environment
    env = "kaggle" if is_kaggle() else "local"
    root = cfg["data"]["roots"][env]

    if env == "kaggle":
        for split in ["train", "val"]:
            cfg["data"][split]["img"] = os.path.join(root, cfg["data"][split]["img"])
            cfg["data"][split]["label"] = os.path.join(root, cfg["data"][split]["label"])

        cfg["training"]["checkpoint_dir"] = "/kaggle/working/checkpoints" # directory will be created by train.py if doesnt' exist

    for split in ["train", "val"]:
        assert os.path.isdir(cfg["data"][split]["img"]), f"{cfg['data'][split]['img']} does not exist"
        assert os.path.isdir(cfg["data"][split]["label"]), f"{cfg['data'][split]['label']} does not exist"

    print(f"Using {env.upper()} data root: {root}")
    return cfg