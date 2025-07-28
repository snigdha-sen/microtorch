## Just an area to try out stuff
import os
import numpy as np
import nibabel as nib
import subprocess
from fit import *


def pull_shapes():
    parent = r"C:\Users\rajib\Documents\GitHub\microtorch\cubric_contributions\test_data"
    bval = os.path.join(parent, "test.bval")
    bvecs = os.path.join(parent, "test.bvec")
    mask = os.path.join(parent, "test_mask.nii.gz")
    img = os.path.join(parent, "test_img.nii.gz")

    bval_data = np.loadtxt(bval)
    bvecs_data = np.loadtxt(bvecs)
    mask_data = nib.load(mask).get_fdata()
    img_data = nib.load(img).get_fdata()

    print(bval_data.shape)
    print(bvecs_data.shape)
    print(mask_data.shape)
    print(img_data.shape)


    args = gen_args_freeform()
    args.image = img
    args.mask = mask
    args.bvals = bval
    args.bvecs = bvecs
    args.model = "SANDI"

    fit_model(args)

    return



if __name__ == "__main__":

    pull_shapes()