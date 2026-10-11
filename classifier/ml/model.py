import torch
import torch.nn as nn
import torchvision.models as models


class OptimizedDermaFusion(nn.Module):
    """
    Optimized DermaFusion model used by MIDAS.

    Image branch:
        EfficientNet-B0 -> 1280 -> 256

    Metadata branch:
        18 features -> 64 -> 64

    Fusion:
        256 + 64 -> 128 -> 2

    Plus:
        visual direct residual branch: 256 -> 2
    """

    def __init__(self, num_meta=18):
        super().__init__()

        # The trained checkpoint contains all EfficientNet weights,
        # so no internet download is required here.
        base = models.efficientnet_b0(weights=None)

        self.features = base.features

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.img_proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.meta_net = nn.Sequential(
            nn.Linear(num_meta, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),

            nn.Linear(128, 2),
        )

        self.visual_direct = nn.Linear(
            256,
            2,
        )

    def forward(self, img, meta):

        features = self.features(img)

        pooled = self.pool(features)

        img_emb = self.img_proj(
            pooled
        )

        meta_emb = self.meta_net(
            meta
        )

        fused = torch.cat(
            [img_emb, meta_emb],
            dim=1,
        )

        logits = (
            self.classifier(fused)
            + self.visual_direct(img_emb)
        )

        return logits