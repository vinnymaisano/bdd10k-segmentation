import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

def extract_dataset():
    '''
    For unpacking the data when dataset is downloaded locally. Not used for Kaggle
    '''
    # find the project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    
    raw_dir = project_root / "data" / "raw"
    target_base = project_root / "data" / "processed"

    # define zips
    zip_files = {
        "images": raw_dir / "bdd100k_images_10k.zip",
        "labels": raw_dir / "bdd100k_seg_maps.zip" 
    }

    # create target base directory
    target_base.mkdir(parents=True, exist_ok=True)

    # process zips
    for category, zip_path in zip_files.items():
        if zip_path.exists():
            print(f"--- extracting {category} ---")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                files = zf.infolist()
                # use tqdm for a progress bar
                for file in tqdm(files, desc=f"unpacking {category}"):
                    zf.extract(file, target_base)
        else:
            print(f"Warning: could not find {zip_path}")

    print(f"\nData extracted to: {target_base}")

    # rename images folder from "10k" to "images"
    source_images = target_base / "10k"
    final_images = target_base / "images"
    if source_images.exists():
        shutil.move(str(source_images), str(final_images))
        print(f"images moved to {final_images}")

if __name__ == "__main__":
    extract_dataset()