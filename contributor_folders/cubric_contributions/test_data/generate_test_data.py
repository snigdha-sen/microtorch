## Just an area to try out stuff
from microtorch.fit import *


bvals_shape = (266,)
bvecs_shape = (3,266)
mask_shape = (110, 110, 66)
img_shape = (110, 110, 66,266)

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
    print(np.unique(bval_data))
    print(np.unique(bvecs_data))



    #args = gen_args()
    #print(args.img)

    bval_redone = transform_numpy_array(bval_data)
    bvecs_redone = transform_numpy_array(bvecs_data)
    img_redone = transform_numpy_array(img_data)
    mask_redone = transform_numpy_array(mask_data)

    print(bval_redone.shape)
    print(bvecs_redone.shape)
    print(mask_redone.shape)
    print(img_redone.shape)

    np.savetxt("test.bval", bval_redone)
    np.savetxt("test.bvec", bvecs_redone)
    nib.save(nib.Nifti1Image(mask_redone, np.eye(4)), "test_mask.nii.gz")
    nib.save(nib.Nifti1Image(img_redone, np.eye(4)), "test_img.nii.gz")

    return


def add_random_values(arr, scale=1.0, distribution='normal'):
    """
    Add random values to a numpy array.

    Parameters:
    -----------
    arr : numpy.ndarray
        Input array to modify
    scale : float, default=1.0
        Scale factor for the random values
    distribution : str, default='normal'
        Type of random distribution ('normal', 'uniform', 'exponential')

    Returns:
    --------
    numpy.ndarray
        Array with random values added
    """
    if distribution == 'normal':
        # Normal distribution with mean=0, std=scale
        random_values = np.random.normal(0, scale, arr.shape)
    elif distribution == 'uniform':
        # Uniform distribution between -scale and +scale
        random_values = np.random.uniform(-scale, scale, arr.shape)
    elif distribution == 'exponential':
        # Exponential distribution, then center around 0
        random_values = np.random.exponential(scale, arr.shape) - scale
    else:
        raise ValueError("Distribution must be 'normal', 'uniform', or 'exponential'")

    return arr + random_values

def transform_numpy_array(input_array):
    # Round all values in the array
    rounded_array = np.round(input_array)

    # Convert all values to their absolute (positive) values
    absolute_array = np.abs(rounded_array)

    return absolute_array

if __name__ == "__main__":
    pull_shapes()