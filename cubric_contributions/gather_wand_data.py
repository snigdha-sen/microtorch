## This file will traverse the WAND dataset and return organised lists of paths - to be used by anyother func to load/train
import torch
from torch.utils.data import Dataset
import os
import glob
import nibabel as nib
import numpy as np

def load_wand(path):
    """
    Load data from a directory structure containing participant and session subfolders.

    Args:
        path (str): Root path containing participant subfolders

    Returns:
        tuple: Four lists containing paths to (images, masks, bvecs, bvals)
               If a file is not found, None is used in its place
    """
    images = []
    masks = []
    bvecs = []
    bvals = []

    # Check if the root path exists
    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist")
        return images, masks, bvecs, bvals

    # Walk through participant directories
    for participant_dir in os.listdir(path):
        participant_path = os.path.join(path, participant_dir)

        # Skip if not a directory
        if not os.path.isdir(participant_path):
            continue

        # Walk through session directories within each participant
        for session_dir in os.listdir(participant_path):
            session_path = os.path.join(participant_path, session_dir)

            # Skip if not a directory
            if not os.path.isdir(session_path):
                continue

            # Look for image file (contains 'noisemap' and ends with '.nii.gz')
            image_files = glob.glob(os.path.join(session_path, "*noisemap*.nii.gz"))
            if image_files:
                images.append(image_files[0])  # Take the first match
            else:
                images.append(None)

            # Look for mask file (contains 'mask' and ends with '.nii.gz')
            mask_files = glob.glob(os.path.join(session_path, "*mask*.nii.gz"))
            if mask_files:
                masks.append(mask_files[0])  # Take the first match
            else:
                masks.append(None)

            # Look for bvecs file (ends with '.bvecs')
            bvecs_files = glob.glob(os.path.join(session_path, "*.bvecs"))
            if bvecs_files:
                bvecs.append(bvecs_files[0])  # Take the first match
            else:
                bvecs.append(None)

            # Look for bvals file (ends with '.bvals')
            bvals_files = glob.glob(os.path.join(session_path, "*.bvals"))
            if bvals_files:
                bvals.append(bvals_files[0])  # Take the first match
            else:
                bvals.append(None)

    print(f"Found {len(images)} sessions")
    print(f"Images: {sum(1 for x in images if x is not None)} found, {sum(1 for x in images if x is None)} missing")
    print(f"Masks: {sum(1 for x in masks if x is not None)} found, {sum(1 for x in masks if x is None)} missing")
    print(f"Bvecs: {sum(1 for x in bvecs if x is not None)} found, {sum(1 for x in bvecs if x is None)} missing")
    print(f"Bvals: {sum(1 for x in bvals if x is not None)} found, {sum(1 for x in bvals if x is None)} missing")



    return images, masks,bvecs,bvals