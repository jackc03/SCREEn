import torch
import torch.nn as nn
import torch.nn.functional as F


# ───── Conv-GRU Cell ───────────────────────────────────────────────────────
class ConvGRUCell(nn.Module):
    def __init__(self, in_c: int, hid_c: int, ks: int = 3):
        super().__init__()
        p = ks // 2
        self.hid_c = hid_c
        self.reset  = nn.Conv2d(in_c + hid_c, hid_c, ks, 1, p)
        self.update = nn.Conv2d(in_c + hid_c, hid_c, ks, 1, p)
        self.out    = nn.Conv2d(in_c + hid_c, hid_c, ks, 1, p)

    def forward(self, x, h):
        if h is None:
            h = x.new_zeros(x.size(0), self.hid_c, x.size(2), x.size(3))
        xc = torch.cat([x, h], 1)
        r = torch.sigmoid(self.reset(xc))
        z = torch.sigmoid(self.update(xc))
        n = torch.tanh(self.out(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


# ───── GRU Stack helper ────────────────────────────────────────────────────
class RecConvStack(nn.Module):
    def __init__(self, in_c: int, hid_list):
        super().__init__()
        cells, prev = [], in_c
        for hid in hid_list:
            cells.append(ConvGRUCell(prev, hid))
            prev = hid
        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        for cell in self.cells:
            x = cell(x, None)
        return x


# ───── Full Network ────────────────────────────────────────────────────────
class SCREEn(nn.Module):
    """
    480 p (854×480) → 1080 p (1920×1080) super-resolution.
    """
    def __init__(self, stem_c: int = 16):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(9, stem_c, 3, 1, 1), nn.ReLU(True)
        )

        self.rec480  = RecConvStack(stem_c, [32, 32, 64])   # was rec720
        self.rec1080 = RecConvStack(64,     [128, 128, 64, 32])

        self.head = nn.Sequential(
            nn.Conv2d(32, 64, 1), nn.ReLU(True),
            nn.Conv2d(64, 32, 1), nn.ReLU(True),
            nn.Conv2d(32, 16, 1), nn.ReLU(True),
            nn.Conv2d(16,  3, 1)
        )

    def forward(self, prev, curr, nxt):
        # concat → (B, 9, 480, 854)
        x = torch.cat([prev, curr, nxt], 1)
        assert x.shape[-2:] == (480, 854), "input must be 854×480"

        feat480 = self.rec480(self.stem(x))                 # (B,32,480,854)

        feat1080 = F.interpolate(feat480, (1080, 1920), mode='bilinear',
                                 align_corners=False)
        feat1080 = self.rec1080(feat1080)                   # (B,16,1080,1920)

        up_ref = F.interpolate(curr, (1080, 1920), mode='bilinear',
                               align_corners=False)

        return (up_ref + self.head(feat1080)).clamp(0, 1)
