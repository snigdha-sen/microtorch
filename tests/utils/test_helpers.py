from pathlib import Path
from microtorch.utils.helpers import strip_filename

def test_strip_filename_nii_gz():
    path = "/data/sub-01/anat/image.nii.gz"
    assert strip_filename(path) == "image"

def test_strip_filename_nii():
    path = "/data/sub-01/anat/image.nii"
    assert strip_filename(path) == "image"
