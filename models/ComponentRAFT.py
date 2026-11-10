                                                                           
import torch
from torch import nn
from core.registry import register_model


@register_model('ComponentRAFT', paper='RAFT-inspired component retrieval', year=2026)
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        l, h = configs.seq_len, configs.pred_len
        self.linear_x = nn.Linear(l, h)
        self.trend = nn.Linear(h, h)
        self.season = nn.Linear(h, h)
        self.fuse = nn.Linear(3*h, h)

    def forward(self, x_enc, components, x_dec=None, x_mark_dec=None, mask=None):
        if not isinstance(components, (tuple, list)) or len(components) not in (2, 4):
            raise ValueError('Expected trend and season references')
        trend, season = components[:2]
        offset = x_enc[:, -1:, :].detach()
        x = self.linear_x((x_enc-offset).transpose(1, 2))
        t = self.trend((trend-offset).transpose(1, 2))
        s = self.season(season.transpose(1, 2))
        y = self.fuse(torch.cat((x, t, s), dim=-1)).transpose(1, 2)
        return y+offset, 0
