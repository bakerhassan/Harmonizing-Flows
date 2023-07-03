from operator import index
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
import warnings
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

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
    def __init__(self, affine_dir, file_path, normalization=None):
        self.normalization = normalization
        self.root_dir = file_path
        self.affine = None
        self.original_shape = None
        self.items = self.get_slices(file_path, affine_dir)

    def get_info(self):
        if self.affine == None or self.original_shape == None:
            raise RuntimeError(f'affine and original shape has to be set before calling this function')
        return self.affine,self.original_shape

    def get_slices(self, file_path, affine_dir):
        slices = []
        affine_matrix = np.load(affine_dir)
        if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
            # Load the NIfTI image
            img = nib.squeeze_image(nib.load(file_path))
            self.original_shape = img.shape
            affine_matrix[:, -1] = img.affine[:, -1]
            self.affine = affine_matrix
            # Resample the image using the given affine transformation
            resampled_img = resample_img(img, target_affine=affine_matrix, interpolation='nearest')

            # Crop the image based on non-empty voxels
            cropped_img = crop_img(resampled_img)

            slices.extend(np.split(cropped_img.get_fdata(), cropped_img.shape[-1], axis=-1))

        return slices

    def transform_volume(self, x):
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
        x = torch.from_numpy(x.transpose((-1, 0, 1)))
        return x

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img = np.squeeze(self.items[index])
        img = transform.resize(img, globals.slice_size, anti_aliasing=True)
        img /= img.max()
        img *= 255
        img = img.astype(np.uint8)
        img = self.transform_volume(img.astype(np.float32))
        return img
