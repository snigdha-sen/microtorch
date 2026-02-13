import numpy as np
import torch

from utils.geometry import sphere2cart, cart2sphere

def test_sphere2cart_z_axis():
    theta = torch.tensor([0.0])
    phi   = torch.tensor([0.0])

    n = sphere2cart(theta, phi)

    expected = torch.tensor([[0.0], [0.0], [1.0]])
    assert torch.allclose(n, expected)

def test_sphere2cart_x_axis():
    theta = torch.tensor([torch.pi / 2])
    phi   = torch.tensor([0.0])

    n = sphere2cart(theta, phi)

    expected = torch.tensor([[1.0], [0.0], [0.0]])
    assert torch.allclose(n, expected, atol=1e-6)

def test_sphere2cart_shape():
    theta = torch.rand(5)
    phi   = torch.rand(5)

    n = sphere2cart(theta, phi)

    assert n.shape == (3, 5)

def test_sphere2cart_unit_norm():
    theta = torch.rand(10)
    phi   = torch.rand(10)

    n = sphere2cart(theta, phi)

    norms = torch.linalg.norm(n, dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

def test_cart2sphere_z_axis():
    xyz = np.array([[0.0, 0.0, 1.0]])

    mu = cart2sphere(xyz)

    theta, phi = mu[0]
    assert np.isclose(theta, 0.0)

def test_cart2sphere_zero_vector():
    xyz = np.array([[0.0, 0.0, 0.0]])

    mu = cart2sphere(xyz)

    assert np.all(mu == 0.0)

def test_sphere_cart_roundtrip():
    theta = torch.rand(10) * (torch.pi - 0.2) + 0.1
    phi   = torch.rand(10) * 2 * torch.pi

    n = sphere2cart(theta, phi).T.numpy()
    mu = cart2sphere(n)

    assert np.allclose(mu[:, 0], theta.numpy(), atol=1e-5)
