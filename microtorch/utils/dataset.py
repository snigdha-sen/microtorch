##This file is for creating custom datasets for qmri models

#As data is quite large, this custom dataset will

import torch
from torch.utils.data import Dataset
import os
import glob
import nibabel as nib
import numpy as np


#Dataset Object will take in a list of images and masks, and a preprocessing function and convert all data into input ready data.
class qMRIDataset(Dataset):

    def __init__(self,
                 images,# list of image paths
                 masks, # list of mask paths
                 preprocess,
                 grad,
                 ):

        super(qMRIDataset, self).__init__()
        ##bvecs + bvals are the same with each image, this dataset could be altered in the future to take individual values for each image. Although this would effect model input

        self.images = images
        self.masks = masks

        self.images_vox = []
        self.masks_vox = []
        self.mask_shapes = []

        self.preprocess = preprocess
        self.grad = grad

        self.preload()

        return

    def preload(self):#This function will load all the data into memory instead of
        for i in range(len(self.images)):
            image_path = self.images[i]
            mask_path = self.masks[i]

            #Load data
            image = torch.from_numpy(nib.load(image_path).get_fdata().astype(np.float32))
            if mask_path is None:
                # No mask provided; use whole image
                mask = torch.ones(image.shape[:3], dtype=torch.float32)
            else:
                # Load mask from file
                mask = torch.from_numpy(nib.load(mask_path).get_fdata().astype(np.float32))

            self.images[i] = image
            self.masks[i] = mask
            self.masks_vox.append(np.shape(mask))

            image_vox, mask_vox = self.preprocess(image, mask)
            self.images_vox.append(image_vox)
            self.mask_shapes.append(mask_vox.shape)

        return

    def __getitem__(self, idx):
        image = self.images_vox[idx]
        mask = self.masks_vox[idx]

        return {
            "image": image,
            "mask": mask
        }

    def __len__(self):

        return len(self.images)





