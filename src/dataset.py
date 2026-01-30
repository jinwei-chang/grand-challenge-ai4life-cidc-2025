import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
import numpy as np

class CalciumDataset(Dataset):
    def __init__(self, tiff_paths, subset, patch_size=(32, 64, 64), samples_per_epoch=1000):
        self.images = []
        self.normals = []
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.subset = subset

        self.base_seed = 42

        for tiff_path in tiff_paths:
            image = tifffile.imread(tiff_path)

            lower, upper = np.percentile(image, (3, 97))

            # self.normals.append((-100.0, 1500.0))
            self.normals.append((lower, upper))
            self.images.append(image)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):

        if self.subset == "valid":
            rng = np.random.default_rng(seed=self.base_seed + idx)
        else:
            rng = np.random.default_rng(None)

        if len(self.patch_size) == 2:
            # image_idx = np.random.randint(len(self.images))
            image_idx = rng.integers(len(self.images))
            image = self.images[image_idx]
            lower, upper = self.normals[image_idx]
            # frame = image[np.random.randint(image.shape[0])]
            frame = image[rng.integers(image.shape[0])]

            h_max, w_max = frame.shape
            # h_start = np.random.randint(h_max - self.patch_size[0])
            # w_start = np.random.randint(w_max - self.patch_size[1])
            h_start = rng.integers(h_max - self.patch_size[0])
            w_start = rng.integers(w_max - self.patch_size[1])

            patch = frame[h_start:h_start+self.patch_size[0], w_start:w_start+self.patch_size[1]]
            patch = np.clip(patch, lower, upper)
            patch = (patch - lower) / (upper - lower + 1e-8)
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()

            masked_patch, mask = self.n2v_mask(patch_tensor, rng)
            return masked_patch, mask, patch_tensor
        elif len(self.patch_size) == 3:
            image_idx = rng.integers(len(self.images))
            image = self.images[image_idx]
            lower, upper = self.normals[image_idx]

            d_max, h_max, w_max = image.shape
            d_start = rng.integers(d_max - self.patch_size[0])
            h_start = rng.integers(h_max - self.patch_size[1])
            w_start = rng.integers(w_max - self.patch_size[2])

            patch = image[d_start:d_start+self.patch_size[0],
                          h_start:h_start+self.patch_size[1],
                          w_start:w_start+self.patch_size[2]]
            patch = np.clip(patch, lower, upper)
            patch = (patch - lower) / (upper - lower + 1e-8)
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()

            masked_patch, mask = self.n2v_mask(patch_tensor, rng)
            return masked_patch, mask, patch_tensor
        else:
            raise ValueError("Invalid patch size")

    def n2v_mask(self, img, rng: np.random.Generator):

        mask_ratio = 0.015

        # rand_tensor = torch.rand(img.shape)
        rand_array = rng.random(size=img.shape, dtype=np.float32)
        rand_tensor = torch.from_numpy(rand_array)
        mask = rand_tensor < mask_ratio

        masked_img = img.clone()

        # fill_values = torch.randn(int(mask.sum())) * img.std() + img.mean()
        # masked_img[mask] = fill_values

        num_masked = int(mask.sum())
        if num_masked > 0:
            # 產生標準常態分佈雜訊
            noise_array = rng.standard_normal(size=num_masked, dtype=np.float32)
            fill_values = torch.from_numpy(noise_array) * img.std() + img.mean()
            masked_img[mask] = fill_values

        return masked_img, mask