import torch
from torch import nn
from monai.networks.nets import ViT


class ViTSegmentation(nn.Module):
    """Matches both notebook ViT implementations (unused hidden_states omitted)."""

    def __init__(
        self,
        in_channels: int = 1,
        img_size: tuple[int, int, int] = (96, 96, 96),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        num_classes: int = 8,
        spatial_dims: int = 3,
    ):
        super().__init__()
        if len(img_size) != 3 or len(patch_size) != 3:
            raise ValueError("3D segmentation requires three-dimensional sizes")
        if any(i <= 0 or p <= 0 or i % p for i, p in zip(img_size, patch_size)):
            raise ValueError("Image dimensions must be divisible by the ViT patch dimensions")
        self.grid_size = tuple(i // p for i, p in zip(img_size, patch_size))
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            classification=False,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose3d(hidden_size, hidden_size // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_size // 2, num_classes, kernel_size=1),
        )
        self.upsample = nn.Upsample(size=img_size, mode="trilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.vit(x)
        b, tokens_count, channels = tokens.shape
        expected_count = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        if tokens_count != expected_count:
            raise ValueError(f"Expected {expected_count} patches but found {tokens_count}")
        volume = tokens.permute(0, 2, 1).reshape(b, channels, *self.grid_size)
        return self.upsample(self.segmentation_head(volume))
