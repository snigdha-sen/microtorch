import numpy as np 
import torch

from microtorch.utils.preprocessing import direction_average, img2voxel, voxel2img, normalise

import torch

class DummyGrad:
    def __init__(self, scheme_matrix, bvalues=None):
        self._scheme_matrix = scheme_matrix

        # Ensure bvalues is always a tensor
        if bvalues is None and scheme_matrix is not None:
            self.bvalues = scheme_matrix[:, -1]  # torch tensor
        elif isinstance(bvalues, tuple):
            self.bvalues = torch.tensor(bvalues, dtype=torch.float32)
        else:
            self.bvalues = bvalues

        # Ensure bvecs is a tensor
        if scheme_matrix is not None:
            self.bvecs = torch.zeros((scheme_matrix.shape[0], 3), dtype=torch.float32)
        else:
            self.bvecs = None

        # Optional fields
        self.TE = None
        self.delta = None
        self.Delta = None
        self.bshape = None
        self.bdelta = None
        self.gradient_strengths = None

        self.number_of_measurements = (
            scheme_matrix.shape[0] if scheme_matrix is not None else None
        )

    def get_scheme_as_matrix(self):
        return self._scheme_matrix

    def set_scheme_from_matrix(self, new_matrix):
        self._scheme_matrix = new_matrix
        self.number_of_measurements = new_matrix.shape[0]

def test_direction_average_basic():
    # img: 2x2x1 voxels, 4 directions
    img = torch.tensor(
        [[[[1.0, 3.0, 10.0, 14.0]]]]
    )  # shape (1,1,1,4)

    # scheme: columns 3+ define shell
    scheme = torch.tensor([
        [0, 0, 0, 1.0],
        [0, 0, 0, 1.0],
        [0, 0, 0, 2.0],
        [0, 0, 0, 2.0],
    ])

    grad = DummyGrad(scheme)

    da_img, da_grad = direction_average(img, grad)

    # Expect two shells
    assert da_img.shape == (1, 1, 1, 2)

    # Means: (1+3)/2 = 2, (10+14)/2 = 12
    assert torch.allclose(da_img[0, 0, 0], torch.tensor([2.0, 12.0]))

def test_img2voxel_basic():
    img = torch.arange(2*2*1*3).float().reshape(2, 2, 1, 3)
    mask = torch.tensor([[[1], [0]], [[1], [0]]])

    X_train, maskvox = img2voxel(img, mask)

    # 4 voxels total, 2 selected
    assert X_train.shape == (2, 3)

    # Check extracted values
    expected = torch.stack([
        img[0, 0, 0],
        img[1, 0, 0],
    ])
    assert torch.allclose(X_train, expected)

def test_voxel2img_roundtrip():
    maskvox = np.array([1, 0, 1, 0])
    params = np.array([10.0, 20.0])
    shape = (2, 2)

    img = voxel2img(params, maskvox, shape)

    expected = np.array([
        [10.0, 0.0],
        [20.0, 0.0],
    ])

    assert np.allclose(img, expected)

def test_normalise_multiple_b0():
    X = torch.tensor([
        [2.0, 4.0, 6.0],
        [1.0, 2.0, 3.0],
    ])

    bvalues = torch.tensor([0.0, 0.0, 1000.0])
    grad = DummyGrad(None, bvalues=bvalues)
    grad.number_of_measurements = 3

    Xn = normalise(X.clone(), grad)

    # mean b0 per voxel
    expected = torch.tensor([
        [2/3, 4/3, 6/3],
        [1/1.5, 2/1.5, 3/1.5],
    ])

    assert torch.allclose(Xn, expected)

def test_normalise_single_b0():
    X = torch.tensor([
        [2.0, 4.0, 6.0],
    ])

    bvalues = torch.tensor([0.0, 1000.0, 1000.0])
    grad = DummyGrad(None, bvalues=bvalues)
    grad.number_of_measurements = 3

    Xn = normalise(X.clone(), grad)

    expected = X / 2.0
    assert torch.allclose(Xn, expected)

