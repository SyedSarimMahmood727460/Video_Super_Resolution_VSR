import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        return x + y

class LiteVSR(nn.Module):
    def __init__(self, scale: int = 2, window: int = 5, base: int = 48, blocks: int = 6):
        super().__init__()
        assert window % 2 == 1
        self.scale = int(scale)
        self.window = int(window)
        self.center = window // 2

        self.enc = nn.Sequential(
            nn.Conv2d(3, base, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base, 3, 1, 1),
            nn.SiLU(inplace=True),
        )

        self.temporal_q = nn.Conv2d(base, base, 1, 1, 0)
        self.temporal_k = nn.Conv2d(base, base, 1, 1, 0)
        self.temporal_v = nn.Conv2d(base, base, 1, 1, 0)

        self.trunk = nn.Sequential(*[ResBlock(base) for _ in range(blocks)])

        up_layers = []
        s = self.scale
        while s > 1:
            if s % 2 == 0:
                up_layers += [
                    nn.Conv2d(base, base * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.SiLU(inplace=True),
                ]
                s //= 2
            elif s % 3 == 0:
                up_layers += [
                    nn.Conv2d(base, base * 9, 3, 1, 1),
                    nn.PixelShuffle(3),
                    nn.SiLU(inplace=True),
                ]
                s //= 3
            else:
                raise ValueError("scale must factor into 2s/3s (e.g., 2,3,4,6,8,9,12)")
        self.upsample = nn.Sequential(*up_layers)
        self.out = nn.Conv2d(base, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x_bt = x.view(b * t, c, h, w)
        feat = self.enc(x_bt).view(b, t, -1, h, w)

        center_feat = feat[:, self.center]
        q = self.temporal_q(center_feat)

        scores = []
        values = []
        for i in range(t):
            k = self.temporal_k(feat[:, i])
            v = self.temporal_v(feat[:, i])
            s = (q * k).sum(dim=1, keepdim=True)
            scores.append(s)
            values.append(v)

        scores = torch.stack(scores, dim=1)
        weights = torch.softmax(scores, dim=1)
        values = torch.stack(values, dim=1)
        fused = (weights * values).sum(dim=1)

        y = self.trunk(fused)
        y = self.upsample(y)
        y = self.out(y)

        center = x[:, self.center]
        center_up = F.interpolate(center, scale_factor=self.scale, mode="bilinear", align_corners=False)
        y = torch.clamp(y + center_up, 0.0, 1.0)
        return y
