                                                                                
import torch
from torch import nn


class TrendSeasonFusion(nn.Module):
    pass                                                                     

                                                                            
                                                                            
                                                              
       

    def __init__(self, horizon, identity_init=False):
        super().__init__()
        self.horizon = horizon
        self.trend_projection = nn.Linear(horizon, horizon)
        self.season_projection = nn.Linear(horizon, horizon)
        self.fusion = nn.Linear(2 * horizon, horizon)
        if identity_init:
            nn.init.zeros_(self.fusion.weight)
            nn.init.zeros_(self.fusion.bias)
            with torch.no_grad():
                self.fusion.weight[:, :horizon].copy_(torch.eye(horizon))

    def forward(self, direct, trend, season):
        retrieved = self.trend_projection(trend) + self.season_projection(season)
        return self.fusion(torch.cat((direct, retrieved), dim=-1))
