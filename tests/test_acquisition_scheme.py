import numpy as np
import pytest
from acquisition_scheme import acquisition_scheme_loader




def test_catch_negative_bvalues(self):
    with pytest.raises(ValueError, match='bvals contains negative values'):
        acquisition_scheme_loader("test_acquisition_scheme.txt")
