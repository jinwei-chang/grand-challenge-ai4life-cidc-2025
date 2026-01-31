import pathlib, os
import tifffile

from dataset import CalciumDataset
from model_arch.unet import UNet
from training import train

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

""" Default or Suggested Settings """
SEED = 42
SLIDE_WINDOW_2D = (16, 16)
SLIDE_WINDOW_3D = (8, 16, 16)
PATCH_SIZE_2D = (64, 64)
PATCH_SIZE_3D = (32, 64, 64)
NMODE_ENABLE = False
MASK_MODE = "gaussian"  # "gaussian", "global_surrounding", "local_surrounding"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    DataLoader 的 worker 初始化函數
    確保每個 worker 擁有獨立但固定的 NumPy/Python seed
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def check_gpu():
    if torch.cuda.is_available():
        # Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"CUDA is available. Number of GPUs: {gpu_count}")
        
        # Iterate over all available GPUs to print details
        for i in range(gpu_count):
            print(f"\n--- GPU Device {i} ---")
            # Get the name of the GPU
            print(f"Device Name: {torch.cuda.get_device_name(i)}")
            
            # Get general properties
            properties = torch.cuda.get_device_properties(i)
            print(f"Total Memory: {round(properties.total_memory / (1024**3), 2)} GB")
            print(f"Multiprocessor Count: {properties.multi_processor_count}")
            print(f"CUDA Capability: {properties.major}.{properties.minor}")
    else:
        print("CUDA is not available.")

    # for MAC GPU
    print("\nChecking for MPS (Apple Silicon GPU) support...")
    if torch.backends.mps.is_available():
        print("MPS (GPU) is available.")
    else:
        print("MPS not available.")


def main():

    check_gpu()

    seed_everything(SEED)
    # 建立 Generator
    g = torch.Generator()
    g.manual_seed(SEED)


    """ temporary parameters """
    PATCH_SIZE = PATCH_SIZE_3D
    MASK_MODE = "gaussian" 
    NMODE_ENABLE = True

    train_path = pathlib.Path(os.getenv("TRAIN_DATA_PATH", "data/train/"))
    valid_path = pathlib.Path(os.getenv("VALID_DATA_PATH", "data/valid/"))
    ground_truth_path = pathlib.Path(os.getenv("GROUND_TRUTH_PATH", "data/valid/F0.tif"))

    train_paths = list(train_path.glob("*.tif"))
    valid_paths = [p for p in valid_path.glob("*.tif") if "f0" not in p.name.lower()]
    valid_paths.sort()
    ground_truth = tifffile.imread(ground_truth_path)

    if PATCH_SIZE == PATCH_SIZE_2D:
        train_dataset = CalciumDataset(train_paths, subset="train", patch_size=PATCH_SIZE, samples_per_epoch=8000, mask_mode=MASK_MODE)
        valid_dataset = CalciumDataset(valid_paths, subset="valid", patch_size=PATCH_SIZE, samples_per_epoch=2000, mask_mode=MASK_MODE)
    elif PATCH_SIZE == PATCH_SIZE_3D:
        train_dataset = CalciumDataset(train_paths, subset="train", patch_size=PATCH_SIZE, samples_per_epoch=8000, mask_mode=MASK_MODE)
        valid_dataset = CalciumDataset(valid_paths, subset="valid", patch_size=PATCH_SIZE, samples_per_epoch=2000, mask_mode=MASK_MODE)
    else:
        raise ValueError("Invalid PATCH_SIZE. Must be either PATCH_SIZE_2D or PATCH_SIZE_3D.")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)

    model = UNet()

    train(model, train_dataset, valid_dataset)

if __name__ == "__main__":
    main()
