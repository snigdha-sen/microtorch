## Just an area to try out stuff
from microtorch.fit import *
from microtorch.utils.args import gen_args
#from cubric_contributions.batch_fit import fit_model_wand

def pull_shapes():
    parent = r"C:\Users\rajib\Documents\GitHub\microtorch\contributor_folders\cubric_contributions\test_data"
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


    args = gen_args()
    args.image = img
    args.mask = mask
    args.bvals = bval
    args.bvecs = bvecs
    args.model = "SANDI"

    fit_model(args)

    #fit_model_single_image_test(img, mask, args)

    #fit_model_wand([img], [mask], args)


    return



if __name__ == "__main__":

    pull_shapes()