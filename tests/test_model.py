import pytest
pytest.importorskip("monai", reason="MONAI must be installed for model integration tests")


def test_ct_vit_output_dimensions_with_lightweight_settings():
    # Full notebook model uses 768 hidden channels and is too big for a tiny CI smoke test.
    from medical_vit.model import ViTSegmentation
    import torch
    net = ViTSegmentation(
        in_channels=1, img_size=(32, 32, 32), patch_size=(16, 16, 16),
        hidden_size=48, mlp_dim=96, num_layers=1, num_heads=3, num_classes=8,
    )
    with torch.no_grad():
        out = net(torch.randn(1, 1, 32, 32, 32))
    assert tuple(out.shape) == (1, 8, 32, 32, 32)


def test_incompatible_patch_size_raises():
    from medical_vit.model import ViTSegmentation
    with pytest.raises(ValueError):
        ViTSegmentation(img_size=(30, 32, 32), patch_size=(16, 16, 16))
