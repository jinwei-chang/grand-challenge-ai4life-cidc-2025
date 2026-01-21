import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
import numpy as np

class CalciumDataset(Dataset):
    def __init__(self, tiff_paths, patch_size=(32, 64, 64), samples_per_epoch=1000):
        self.images = []
        self.normals = []
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch

        for tiff_path in tiff_paths:
            image = tifffile.imread(tiff_path)

            self.normals.append((-100.0, 1500.0))
            self.images.append(image)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if len(self.patch_size) == 2:
            image_idx = np.random.randint(len(self.images))
            image = self.images[image_idx]
            lower, upper = self.normals[image_idx]
            frame = image[np.random.randint(image.shape[0])]

            h_max, w_max = frame.shape
            h_start = np.random.randint(h_max - self.patch_size[0])
            w_start = np.random.randint(w_max - self.patch_size[1])

            patch = frame[h_start:h_start+self.patch_size[0], w_start:w_start+self.patch_size[1]]
            patch = np.clip(patch, lower, upper)
            patch = (patch - lower) / (upper - lower + 1e-8)
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()

            masked_patch, mask = self.n2v_mask(patch_tensor)
            return masked_patch, mask, patch_tensor
        elif len(self.patch_size) == 3:
            raise NotImplementedError("3D patches not implemented")
        else:
            raise ValueError("Invalid patch size")

    def n2v_mask(self, img):

        mask_ratio = 0.015

        rand_tensor = torch.rand(img.shape)
        mask = rand_tensor < mask_ratio

        masked_img = img.clone()
        fill_values = torch.randn(int(mask.sum())) * img.std() + img.mean()
        masked_img[mask] = fill_values

        return masked_img, mask