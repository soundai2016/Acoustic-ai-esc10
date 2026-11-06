from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False, image_size: int = 224):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
        except (ImportError, RuntimeError) as exc:
            raise ImportError('ResNet-18 requires matching torch and torchvision builds') from exc
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            self.backbone = resnet18(weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError('Could not load ImageNet pretrained ResNet-18 weights. Check network access or use --no-pretrained.') from exc
            raise
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.image_size = image_size
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        self.register_buffer('image_mean', mean)
        self.register_buffer('image_std', std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        # Map standardized log-mel values into a bounded image-like range.
        x = ((x.clamp(-3.0, 3.0) + 3.0) / 6.0).repeat(1, 3, 1, 1)
        x = (x - self.image_mean) / self.image_std
        return self.backbone(x)

    def freeze_backbone(self, frozen: bool) -> None:
        for name, param in self.backbone.named_parameters():
            param.requires_grad = (name.startswith('fc.') or not frozen)
        if frozen:
            self.backbone.eval()
            self.backbone.fc.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and not any(p.requires_grad for p in self.backbone.layer1.parameters()):
            self.backbone.eval()
            self.backbone.fc.train()
        return self


def build_model(name: str, num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    if name == 'resnet18':
        return AudioResNet18(num_classes, pretrained=pretrained)
    if name == 'tiny':
        if pretrained:
            raise ValueError('Pretraining is not available for TinyCNN')
        return TinyCNN(num_classes)
    raise ValueError(f'Unknown architecture: {name}')


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
