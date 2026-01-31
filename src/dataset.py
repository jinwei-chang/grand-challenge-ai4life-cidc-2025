import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
import numpy as np

class CalciumDataset(Dataset):
    def __init__(self, tiff_paths, subset, patch_size=(32, 64, 64), samples_per_epoch=1000, mask_mode="gaussian"):
        self.images = []
        self.normals = []
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.subset = subset

        self.base_seed = 42

        self.mask_mode = mask_mode

        for tiff_path in tiff_paths:
            image = tifffile.imread(tiff_path)

            # lower, upper = np.percentile(image, (1, 99))

            # self.normals.append((-100.0, 1500.0))
            # self.normals.append((lower, upper))
            self.images.append(image)

        global_normals = np.percentile(np.concatenate(self.images), (1, 99.9))
        # print(f"Global normalization percentiles: {global_normals}")

        for _ in range(len(tiff_paths)):
            """ method 1: Use fixed normalization values for all images """
            # self.normals.append((-100.0, 3000.0))

            """ method 2: Use global normalization values for all images """
            self.normals.append((global_normals[0], global_normals[1]))

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

            """ Apply data augmentation """
            patch_tensor = self.apply_augmentation(patch_tensor, rng)

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

            """ Apply data augmentation """
            patch_tensor = self.apply_augmentation(patch_tensor, rng)

            masked_patch, mask = self.n2v_mask(patch_tensor, rng)
            
            return masked_patch, mask, patch_tensor
        else:
            raise ValueError("Invalid patch size")

    def apply_augmentation(self, patch, rng: np.random.Generator):


        """ Brightness adjustment """
        min_scale, max_scale = 0.6, 1.4
        scale_factor = rng.uniform(min_scale, max_scale)
        patch_aug = patch * scale_factor

        return patch_aug


    def n2v_mask(self, img, rng: np.random.Generator):

        mask_ratio = 0.015
        rand_array = rng.random(size=img.shape, dtype=np.float32)
        rand_tensor = torch.from_numpy(rand_array)
        mask = rand_tensor < mask_ratio

        masked_img = img.clone()

        num_masked = int(mask.sum())
        if num_masked == 0:
            return masked_img, mask

        if self.mask_mode == "gaussian":
            noise_array = rng.standard_normal(size=num_masked, dtype=np.float32)
            fill_values = torch.from_numpy(noise_array) * img.std() + img.mean()
            masked_img[mask] = fill_values
        elif self.mask_mode == "global_surrounding":
            # 1. 找出所有 "沒有被 Mask" 的健康像素座標
            # 注意：如果 Patch 很大，這步可能會有點慢，但在 Patch Size 下通常沒問題
            unmasked_indices = torch.nonzero(~mask, as_tuple=False)
            
            # 2. 從健康像素中隨機抽樣
            # rng.choice 在一維陣列上操作，所以我們隨機選 index 的 index
            rand_select_idx = rng.integers(0, unmasked_indices.shape[0], size=num_masked)
            
            # 3. 取出對應的座標
            selected_coords = unmasked_indices[rand_select_idx]
            
            # 4. 根據維度取值 (支援 2D/3D)
            ndim = img.ndim
            coords_tuple = tuple(selected_coords[:, i] for i in range(ndim))
            fill_values = img[coords_tuple]
            
            masked_img[mask] = fill_values
        elif self.mask_mode == "local_surrounding":
            radius = 2 # 5x5 window
            masked_indices = torch.nonzero(mask, as_tuple=False)
            ndim = img.ndim

            # 1. 產生隨機位移 [-radius, radius]
            offsets = rng.integers(-radius, radius + 1, size=masked_indices.shape)
            offsets = torch.from_numpy(offsets).to(img.device)
            
            # 2. 【關鍵修正】鎖定時間軸 (如果是 3D)
            # 假設 img shape 是 (C, D, H, W) -> ndim=4
            # 或者是 (D, H, W) 假如沒有 Channel -> ndim=3
            # 我們假設第 0 維是 Channel，不應該動。
            # 如果是 3D Patch (Channel, Depth/Time, Height, Width)
            # 我們應該鎖定 dim=1 (Time)，只允許 dim=2,3 (H, W) 移動
            
            # 為了通用性，我們把所有非 Spatial 的維度偏移都設為 0
            # 通常最後兩維是 H, W
            spatial_dims = 2 
            non_spatial_dims = ndim - spatial_dims # 前面幾個維度 (C, D)
            
            if non_spatial_dims > 0:
                offsets[:, :non_spatial_dims] = 0 # 鎖定 C 和 D(Time)
            
            # 3. 計算來源座標
            source_indices = masked_indices + offsets
            
            # 4. 邊界檢查 (Clamp)
            for d in range(ndim):
                dim_size = img.shape[d]
                source_indices[:, d].clamp_(0, dim_size - 1)
            
            # 5. 填補
            indices_tuple = tuple(source_indices[:, i] for i in range(ndim))
            fill_values = img[indices_tuple]
            masked_img[mask] = fill_values

        return masked_img, mask