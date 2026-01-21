import pathlib, os
import tifffile

from dataset import CalciumDataset
from models.unet import UNet
from training import train

import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    """ temporary parameters """
    PATCH_SIZE_2D = (64, 64)
    PATCH_SIZE_3D = (32, 64, 64)
    PATCH_SIZE = PATCH_SIZE_2D

    train_path = pathlib.Path(os.getenv("TRAIN_DATA_PATH", "data/train/"))
    valid_path = pathlib.Path(os.getenv("VALID_DATA_PATH", "data/valid/"))
    ground_truth_path = pathlib.Path(os.getenv("GROUND_TRUTH_PATH", "data/valid/F0.tif"))

    train_paths = list(train_path.glob("*.tif"))
    valid_paths = [p for p in valid_path.glob("*.tif") if "f0" not in p.name.lower()]
    ground_truth = tifffile.imread(ground_truth_path)

    train_dataset = CalciumDataset(train_paths, patch_size=PATCH_SIZE, samples_per_epoch=8000)
    valid_dataset = CalciumDataset(valid_paths, patch_size=PATCH_SIZE, samples_per_epoch=2000)

    model = UNet()

    train(model, train_dataset, valid_dataset)

if __name__ == "__main__":
    main()
