                                                                          
import torch
from torch import nn
from models.TrendSeasonFusion import TrendSeasonFusion


class ComponentForecastAdapter(nn.Module):
    def __init__(self, backbone, horizon, centered_references=False):
        super().__init__()
        self.backbone = backbone
        self.centered_references = centered_references
        self.fusion = TrendSeasonFusion(horizon, identity_init=True)

    def forward(self, x, components, x_mark=None):
        forecast, additional_loss = self.backbone(x, x_mark, None, None)
        trend, season = components[:2]
        mean = x.mean(dim=1, keepdim=True).detach()
        scale = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        if self.centered_references:
            trend = trend / scale
        else:
            trend = (trend - mean) / scale
        season = season / scale
        direct = ((forecast - mean) / scale).transpose(1, 2)
        fused = self.fusion(direct, trend.transpose(1, 2), season.transpose(1, 2))
        return fused.transpose(1, 2) * scale + mean, additional_loss
