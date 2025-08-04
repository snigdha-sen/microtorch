##This file is for creating custom datasets for qmri models

#As data is quite large, this custom dataset will

import torch
from torch.utils.data import Dataset
import os
import glob
import nibabel as nib
import numpy as np

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
        self.preprocess = preprocess
        self.grad = grad
        self.preloaded = False
        return

    def preload(self):#This function will load all the data into memory instead of
        return

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        #Load data
        image = torch.from_numpy(nib.load(image_path).get_fdata().astype(np.float32))
        if mask_path is None:
            # No mask provided; use whole image
            mask = torch.ones(image.shape[:3], dtype=torch.float32)
        else:
            # Load mask from file
            mask = torch.from_numpy(nib.load(mask_path).get_fdata().astype(np.float32))
        #Preprocess data

        if self.preprocess is not None:
            image, mask = self.preprocess(image, mask)
        #Return data

        return {
            "image": image,
            "mask": mask
        }

    def __len__(self):

        return len(self.images)





