import argparse
import random
import os

#Effectively using this this version of the args generator as a kind of config framework - works quite well!
def gen_args(): ### This removes the required flags so we can generate this and add arguements during run time
    parser = argparse.ArgumentParser()
    parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=20)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=3e-4)
    parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
    parser.add_argument("-f", "--folder", help="Folder where image & mask are stored", default=os.getcwd())
    parser.add_argument("-img", "--image", help="Path of the image to train on")
    parser.add_argument("-ma", "--mask", help="Path of the mask to apply to image", default=None, type=str)
    parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=256)
    parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=3)
    parser.add_argument("-m", "--model", type=str,
                        help="Compartmental Model to use. Implemented are verdict, sandi, or user defined ones form combinations of ball; sphere, stick; astrosticks; cylinder; astrocylinders; zeppelin; astrozeppelins; dot.",
                        default="verdict")
    parser.add_argument("-a", "--activation", type=str,
                        help="Activation function to use with mlp: elu, relu, prelu or tanh.", default="prelu")
    parser.add_argument("-op", "--operation", help="Operation to perform (train+fit, train, fit).", default="train+fit")
    parser.add_argument("-bvals", "--bvals", help="bvals file in FSL format and in [s/mm2]", default=None, type=str)
    parser.add_argument("-bvecs", "--bvecs", help="bvecs file in FSL format", default=None, type=str)
    parser.add_argument("-grad", "--grad", help="acquisition scheme file in FSL format and in [s/mm2]", default=None,
                        type=str)
    parser.add_argument("-d", "--delta", help="txt file with gradient pulse separation (ms)",
                        default=23.7)
    parser.add_argument("-sd", "--smalldelta", help="txt file with gradient pulse duration (ms)",
                        default=6)
    parser.add_argument("-TE", "--TE", help="echo time in ms", default=None)
    parser.add_argument("-TR", "--TR", help="repetition time in ms", default=None)
    parser.add_argument("-TI", "--TI", help="inversion time in ms", default=None)
    parser.add_argument("-df", "--dropout_frac", help="dropout fraction", type=float, default=0.2)
    parser.add_argument("-lmax", "--lmax", help="max order used for spherical harmonics", default=2)
    parser.add_argument("-bd", "--bdelta", help="shape of gradient pulse", default=1, type=float)
    parser.add_argument("-c", "--clip", type=str,
                        help="Clipping method to go to parameter space. Options are clamp and sigmoid", default="clamp")

    args = parser.parse_args()

    return args