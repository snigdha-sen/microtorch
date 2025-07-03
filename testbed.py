## Just an area to try out stuff
import os
import numpy as np
import nibabel as nib
import subprocess
from fit import *


def pull_shapes():

    fit = r"/home/c21025395/PycharmProjects/microtorch/fit.py"
    parent = r"/cubric/collab/314_wand/diffusion/3TM_updatedSANDI_em/derivatives/preprocessed/sub-31400395/ses-01"
    bval = os.path.join(parent, "314_00395_CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift-preproc_dwi.bval")
    bvecs = os.path.join(parent,"314_00395_CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift-preproc_dwi.bvec")
    mask = os.path.join(parent,"314_00395_CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift-preproc_mask.nii.gz")
    img = os.path.join(parent,"314_00395_CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift-preproc_dwi.nii.gz")

    bval_data = np.loadtxt(bval)
    bvecs_data = np.loadtxt(bvecs)
    mask_data = nib.load(mask).get_fdata()
    img_data = nib.load(img).get_fdata()

    print(bval_data.shape)
    print(bvecs_data.shape)
    print(mask_data.shape)
    print(img_data.shape)

    args = gen_args()
    print(args.img)
    

    

    return



if __name__ == "__main__":

    pull_shapes()