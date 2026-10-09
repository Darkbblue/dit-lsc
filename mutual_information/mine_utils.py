import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class VectorizeProject(nn.Module):
    """
    Projects arbitrary tensors (2D [B,D] or 4D [B,C,H,W]) to a fixed vector dim.
    - For 4D inputs: 1x1 conv -> GELU -> AdaptiveAvgPool2d(1) -> flatten -> Linear.
    - For 2D inputs: MLP.
    """
    def __init__(self, out_dim: int = 256, mid_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        # 4D path
        self.conv = nn.Conv2d(in_channels=1, out_channels=mid_dim, kernel_size=1)  # dummy init; reset per input
        self.conv_act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc4d = nn.Linear(mid_dim, out_dim)

        # 2D path
        self.fc2d = nn.Sequential(
            nn.Linear(1, mid_dim),  # dummy init; reset per input
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
        )

        self.dev = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    def _reset_for_4d(self, C: int):
        if self.conv.in_channels != C:
            self.conv = nn.Conv2d(C, self.conv.out_channels, kernel_size=1).to(self.dev)

    def _reset_for_2d(self, D: int):
        if isinstance(self.fc2d[0], nn.Linear) and self.fc2d[0].in_features != D:
            self.fc2d = nn.Sequential(
                nn.Linear(D, self.fc2d[0].out_features),
                nn.GELU(),
                nn.Linear(self.fc2d[0].out_features, self.out_dim),
            ).to(self.dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,D] or [B,C,H,W]
        if x.dim() == 4:
            B, C, H, W = x.shape
            self._reset_for_4d(C)
            h = self.conv_act(self.conv(x))     # [B, mid_dim, H, W]
            h = self.pool(h).squeeze(-1).squeeze(-1)  # [B, mid_dim]
            z = self.fc4d(h)                    # [B, out_dim]
            return z
        elif x.dim() == 2:
            B, D = x.shape
            self._reset_for_2d(D)
            return self.fc2d(x)                 # [B, out_dim]
        else:
            raise ValueError(f"Unsupported tensor rank {x.dim()} (expected 2D or 4D).")

def make_negatives(y: torch.Tensor) -> torch.Tensor:
    """Shuffle batch to break X-Y pairing (negative samples for MINE)."""
    idx = torch.randperm(y.size(0), device=y.device)
    return y[idx]

def concat_joint(n_vec: torch.Tensor, t_vec: torch.Tensor) -> torch.Tensor:
    """Concatenate noise and target embeddings to estimate I(X; [N,T])."""
    return torch.cat([n_vec, t_vec], dim=1)
