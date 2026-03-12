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
    


    
def most_recent_output_file(output_root, model_name):
    """
    Find the most recent output file for a given model.

    Args:
        output_root (str or Path): Path to the outputs directory.
        model_name (str): Name of the model (e.g., 'VERDICT', 'BallStick').

    Returns:
        Path or None: Path to the most recent file matching the model, or None if not found.
    """
    output_root = Path(output_root)

    # Pattern: model_name*_param_maps.nii.gz
    pattern = f"{model_name}*_param_maps.nii.gz"

    # Recursively search for matching files
    files = list(output_root.rglob(pattern))
    if not files:
        return None

    # Sort by modification time (most recent first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0]