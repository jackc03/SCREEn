import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────── Conv-GRU Cell ───────────────────────────────────────────
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


# ────────────────── GRU Stack helper ────────────────────────────────────────
class RecConvStack(nn.Module):
    def __init__(self, in_c: int, hid_list):
        super().__init__()
        layers, prev = [], in_c
        for hid in hid_list:
            layers.append(ConvGRUCell(prev, hid))
            prev = hid
        self.cells = nn.ModuleList(layers)

    def forward(self, x):
        for cell in self.cells:
            x = cell(x, None)      # no temporal loop ⇒ h=None each call
        return x


# ────────────────── Full Network ────────────────────────────────────────────
class SCREEn(nn.Module):
    def __init__(self, stem_c: int = 16):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(9, stem_c, 3, 1, 1),
            nn.ReLU(True)
        )

        self.rec720  = RecConvStack(stem_c, [32, 32, 32])
        self.rec1080 = RecConvStack(32,     [32, 32, 16, 16])

        self.head = nn.Sequential(
            nn.Conv2d(16, 32, 1), nn.ReLU(True),
            nn.Conv2d(32, 16, 1), nn.ReLU(True),
            nn.Conv2d(16,  3, 1)
        )

    # ---------------------- forward with assertions -------------------------
    def forward(self, prev, curr, nxt):
        x = torch.cat([prev, curr, nxt], 1)               # (B,9,720,1280)
        assert x.shape[1] == 9, "concatenation produced wrong channel count"

        feat720 = self.rec720(self.stem(x))               # (B,32,720,1280)
        assert feat720.shape[1] == 32, "rec720 output ≠ 32 channels"

        feat1080 = F.interpolate(feat720, (1080, 1920), mode='bilinear',
                                 align_corners=False)
        assert feat1080.shape[1] == 32, "upsample kept wrong channel count"

        feat1080 = self.rec1080(feat1080)                 # (B,16,1080,1920)
        assert feat1080.shape[1] == 16, "rec1080 output ≠ 16 channels"

        up_ref = F.interpolate(curr, (1080, 1920), mode='bilinear',
                               align_corners=False)

        out = up_ref + self.head(feat1080)
        return out.clamp(0.0, 1.0)
