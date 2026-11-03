from dataclasses import replace
from pathlib import Path
import pytest
from medical_vit.config import load_config

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("modality,shape", [("ct", (96, 96, 96)), ("mri", (128, 128, 96))])
def test_yaml_configuration(modality, shape):
    config = load_config(ROOT / "configs" / f"{modality}.yaml")
    assert config.modality == modality
    assert config.img_size == shape
    assert config.nfolds == 5
    config.validate()

def test_unknown_option_is_rejected(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("modality: ct\nunused_setting: true\n")
    with pytest.raises(ValueError, match="Unknown config"):
        load_config(config)
