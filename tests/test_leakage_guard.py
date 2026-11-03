import pytest

monai = pytest.importorskip("monai")
from medical_vit.data import _paths

def test_split_overlap_is_detectable():
    training = [{"image": "case_001.nii.gz"}, {"image": "case_002.nii.gz"}]
    validation = [{"image": "case_003.nii.gz"}]
    assert not (_paths(training) & _paths(validation))
    assert _paths(training) & _paths([{"image": "case_002.nii.gz"}])
