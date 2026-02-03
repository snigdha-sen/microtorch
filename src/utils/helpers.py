from pathlib import Path

def strip_filename(path):

    """
    Removes folder path and strips '.nii' or '.nii.gz' extension from a file path.

    Parameters:
        path (str or Path): Full file path.

    Returns:
        str: Filename without path and extension.
    """
    path = Path(path)
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    elif name.endswith(".nii"):
        return name[:-4]
    else:
        return path.stem