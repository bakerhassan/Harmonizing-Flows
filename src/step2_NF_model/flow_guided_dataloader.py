import torch
import numpy as np
from torch.utils.data import Dataset
import imgaug.augmenters as iaa

import os
import nibabel as nib
from nilearn.image import resample_img, crop_img
from skimage import transform
from src.globals import globals
import warnings

warnings.filterwarnings("ignore")


class MedicalImage2DDataset(Dataset):

    # TODO this code largely intersects with the file in step1_Harmonizer_network package.
    def get_slices(self, folder_path, affine_dir):
        slices = []
        affine_matrix = np.load(affine_dir)
        # Loop over NIfTI files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                file_path = os.path.join(folder_path, filename)

                # Load the NIfTI image
                img = nib.squeeze_image(nib.load(file_path))
                affine_matrix[:, -1] = img.affine[:, -1]
                # Resample the image using the given affine transformation
                resampled_img = resample_img(img, target_affine=affine_matrix, interpolation='nearest')

                # Crop the image based on non-empty voxels
                cropped_img = crop_img(resampled_img)

                slices.extend(np.split(cropped_img.get_fdata(), cropped_img.shape[-1], axis=-1))

        return slices

    def __init__(self, mode, affine_dir, root_dir, normalization=None):
        self.normalization = normalization
        self.root_dir = root_dir
        self.mode = mode
        self.items = self.get_slices(root_dir, affine_dir)
        self.aug = iaa.Sequential(
            [
                iaa.Sometimes(0.6, iaa.Add((-30, 30))),
                iaa.Sometimes(0.6, iaa.Multiply((0.5, 1.5))),
                iaa.Sometimes(0.6, iaa.GammaContrast((0.5, 1.5)))
            ])

    def transform_volume(self, x):
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
        x = torch.from_numpy(x.transpose((-1, 0, 1)))
        return x

    def __len__(self):
        return len(self.items)

    def new_map(self):
        points = {}
        for i in range(257):
            points[i] = None
        points[0] = 0
        points[256] = 255
        for level in range(8):
            interval = 2 ** (8 - level - 1)
            for point in range(0, 256, interval):
                if points[point] is None:
                    dis = 0.2 * (points[point + interval] - points[point - interval])
                    points[point] = np.random.randint(int(points[point - interval] + dis),
                                                      int(points[point + interval] - dis) + 1)
        return np.array(list(points.values()))

    def __getitem__(self, index):
        img = np.squeeze(self.items[index])
        img = transform.resize(img, globals.slice_size, anti_aliasing=True)
        img /= img.max()
        img *= 255
        img = img.astype(np.uint8)
        if self.mode == 'train':
            if np.random.random() < 0.2:
                rnd = np.random.random()
                if rnd < 0.5:
                    img2 = self.new_map()[img]
                else:
                    img2 = self.aug(image=img)
                    img2 = np.clip(img2, 0, 255).astype(np.uint8)
                if (np.mean(np.abs(img - img2)) + 2 * np.std(img - img2)) > 20:
                    img2 = self.transform_volume(img2.astype(np.float32))
                    return [img2, -1]
        elif self.mode == 'val':
            if index % 5 == 0:
                rnd = np.random.random()
                if rnd < 0.5:
                    img2 = self.new_map()[img]
                else:
                    img2 = self.aug(image=img)
                    img2 = np.clip(img2, 0, 255).astype(np.float32)
                while (np.mean(np.abs(img - img2)) + 2 * np.std(img - img2)) < 20:
                    rnd = np.random.random()
                    if rnd < 0.5:
                        img2 = self.new_map()[img]
                    else:
                        img2 = self.aug(image=img)
                        img2 = np.clip(img2, 0, 255).astype(np.float32)
                img2 = self.transform_volume(img2.astype(np.float32))
                return [img2, -1]
        img = self.transform_volume(img.astype(np.float32))
        return [img, 1]
