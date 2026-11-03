import pytest
from medical_vit.config import get_config


def test_ct_reproduces_source_preprocessing_configuration():
    ct = get_config("ct")
    assert ct.dataset_filename == "dataset_1.json"
    assert ct.img_size == (96, 96, 96)
    assert (ct.intensity_min, ct.intensity_max) == (-150.0, 1113.0)
    assert ct.include_background is False
    ct.validate()


def test_mri_reproduces_source_preprocessing_configuration():
    mri = get_config("mri")
    assert mri.dataset_filename == "dataset_0.json"
    assert mri.img_size == (128, 128, 96)
    assert (mri.intensity_min, mri.intensity_max) == (6.0, 1200.0)
    assert mri.include_background is True
    mri.validate()


def test_unknown_modality_fails():
    with pytest.raises(ValueError):
        get_config("pet")
