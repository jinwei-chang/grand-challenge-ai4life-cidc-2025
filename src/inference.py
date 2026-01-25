import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denoise_frame(model, frame, device, patch_size, slide_window=16, batch_size=16):
    model.eval()
    if len(patch_size) != 2:
        raise ValueError("Patch size must be a tuple of length 2")

    frame = frame.unsqueeze(0).unsqueeze(0).float()

    padding_frame = F.pad(frame, (slide_window, slide_window, slide_window, slide_window), mode='reflect')
    padding_frame = padding_frame.to(device)

    output_sum = torch.zeros_like(padding_frame)
    count_map = torch.zeros_like(padding_frame)

    batch_patches = []
    batch_coords = []

    _, _, h_pad, w_pad = padding_frame.shape

    with torch.no_grad():
        denoised_frame = torch.zeros_like(padding_frame)
        for h in range(0, padding_frame.shape[2] - patch_size[0] + 1, slide_window):
            for w in range(0, padding_frame.shape[3] - patch_size[1] + 1, slide_window):
                patch = padding_frame[:, :, h:h+patch_size[0], w:w+patch_size[1]]
                batch_patches.append(patch)
                batch_coords.append((h, w))

                if len(batch_patches) == batch_size:
                    batch_patches = torch.cat(batch_patches, dim=0)
                    denoised_batch = model(batch_patches)
                    for i, (ph, pw) in enumerate(batch_coords):
                        output_sum[:,:,ph:ph+patch_size[0], pw:pw+patch_size[1]] += denoised_batch[i]
                        count_map[:,:,ph:ph+patch_size[0], pw:pw+patch_size[1]] += 1
                    batch_patches = []
                    batch_coords = []
                # denoised_patch = model(patch)
                # output_sum[:, :, h:h+patch_size[0], w:w+patch_size[1]] += denoised_patch
                # count_map[:, :, h:h+patch_size[0], w:w+patch_size[1]] += 1
                # denoised_frame[:,:,h:h+patch_size[0], w:w+patch_size[1]] = denoised_patch
        if len(batch_patches) > 0:
            batch_patches = torch.cat(batch_patches, dim=0)
            denoised_batch = model(batch_patches)
            for i, (ph, pw) in enumerate(batch_coords):
                output_sum[:,:,ph:ph+patch_size[0], pw:pw+patch_size[1]] += denoised_batch[i]
                count_map[:,:,ph:ph+patch_size[0], pw:pw+patch_size[1]] += 1

        count_map[count_map == 0] = 1.0
        denoised_frame = output_sum / count_map
        denoised_frame = denoised_frame[:, :, slide_window:-slide_window, slide_window:-slide_window]
    return denoised_frame


def denoise_volume(model, volume, patch_size, slide_window=8, batch_size=4):
    if len(patch_size) != 3:
        raise ValueError("Patch size must be a tuple of length 3 for 3D denoising")  

    volume = volume.unsqueeze(0).unsqueeze(0).float()
    volume = volume.to(device)

    padding_volume = F.pad(volume, (slide_window, slide_window, slide_window, slide_window, slide_window, slide_window), mode='reflect')

    output_sum = torch.zeros_like(padding_volume)
    count_map = torch.zeros_like(padding_volume)

    batch_patches = []
    batch_coords = []

    _, _, d_pad, h_pad, w_pad = padding_volume.shape

    with torch.no_grad():
        for d in range(0, d_pad - patch_size[0] + 1, slide_window):
            for h in range(0, h_pad - patch_size[1] + 1, slide_window):
                for w in range(0, w_pad - patch_size[2] + 1, slide_window):
                    patch = padding_volume[:, :, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]]
                    batch_patches.append(patch)
                    batch_coords.append((d, h, w))

                    if len(batch_patches) == batch_size:
                        batch_patches = torch.cat(batch_patches, dim=0)
                        denoised_batch = model(batch_patches)
                        for i, (pd, ph, pw) in enumerate(batch_coords):
                            output_sum[:,:,pd:pd+patch_size[0], ph:ph+patch_size[1], pw:pw+patch_size[2]] += denoised_batch[i]
                            count_map[:,:,pd:pd+patch_size[0], ph:ph+patch_size[1], pw:pw+patch_size[2]] += 1
                        batch_patches = []
                        batch_coords = []
        
        if len(batch_patches) > 0:
            batch_patches = torch.cat(batch_patches, dim=0)
            denoised_batch = model(batch_patches)
            for i, (pd, ph, pw) in enumerate(batch_coords):
                output_sum[:,:,pd:pd+patch_size[0], ph:ph+patch_size[1], pw:pw+patch_size[2]] += denoised_batch[i]
                count_map[:,:,pd:pd+patch_size[0], ph:ph+patch_size[1], pw:pw+patch_size[2]] += 1

        count_map[count_map == 0] = 1.0
        denoised_volume = output_sum / count_map
        denoised_volume = denoised_volume[:, :, slide_window:-slide_window, slide_window:-slide_window, slide_window:-slide_window]
    return denoised_volume

def denoise_video(model, video, patch_size, slide_window=16, batch_size=16):
    model.eval()
    model.to(device)

    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)
    
    video = video.to(device)
    
    
    if len(patch_size) == 2:
        denoised_frames = torch.zeros_like(video)
        loop = tqdm(range(video.shape[0]), leave=True)
        for t in loop:
            frame = video[t]
            denoised_frame = denoise_frame(model, frame, device, patch_size, slide_window=slide_window, batch_size=batch_size)
            denoised_frames[t] = denoised_frame.squeeze()
        return denoised_frames
    elif len(patch_size) == 3:
        denoised_frames = torch.zeros_like(video)
        loop = tqdm(range(video.shape[0] // patch_size[0]), leave=True)
        for t in loop:
            start_idx = t * patch_size[0]
            end_idx = start_idx + patch_size[0]
            volume_part = video[start_idx:end_idx]
            denoised_volume_part = denoise_volume(model, volume_part, patch_size, slide_window=slide_window, batch_size=batch_size)
            denoised_frames[start_idx:end_idx] = denoised_volume_part.squeeze()

        return denoised_frames
        # part_frames = denoised_frames
        # return denoise_volume(model, video, patch_size, slide_window=slide_window, batch_size=batch_size).squeeze()
    else:
        raise ValueError("Invalid patch size")
