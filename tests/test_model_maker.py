import pytest
# Test class to test comp mean handling.
class DummyComp:
    def __init__(self, mean):
        self.spherical_mean = mean


class DummyContainer:
    def __init__(self, comps):
        self.comps = comps
        if len({comp.spherical_mean for comp in comps if comp.spherical_mean is not None}) != 1:
            raise ValueError(
                "Invalid input. All compartments must have the same spherical mean property, either all spherical mean or all not spherical mean."
            )


# Should not raise error for these

def test_all_mean_true():
    comps = [DummyComp(True) for _ in range(3)]
    DummyContainer(comps)


def test_all_mean_false():
    comps = [DummyComp(False) for _ in range(3)]
    DummyContainer(comps)


def test_two_true_one_none():
    comps = [DummyComp(True), DummyComp(True), DummyComp(None)]
    DummyContainer(comps)


def test_two_false_one_none():
    comps = [DummyComp(False), DummyComp(False), DummyComp(None)]
    DummyContainer(comps)

# Should raise errors
@pytest.mark.parametrize(
    "comps",
    [
        [DummyComp(True), DummyComp(False)], # Mix spherical mean
        [DummyComp(True), DummyComp(None), DummyComp(False)] # Mix plus none
    ]
)
def test_mixed_mean_raise(comps):
    with pytest.raises(ValueError, match="Invalid input"):
        DummyContainer(comps)