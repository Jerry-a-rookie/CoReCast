                                                                         
import torch
from torch import nn
from core.registry import register_model
from models.TrendSeasonFusion import TrendSeasonFusion


@register_model(
    'ComponentRAFTSeasonTrend',
    paper='RAFT-inspired trend-season retrieval forecasting',
    year=2026,
)
class Model(nn.Module):
    pass                                                                    

                                                                            
                                                                       
                                                                          
                                                                            
       

    def __init__(self, configs):
        super().__init__()
        seq_len, pred_len = configs.seq_len, configs.pred_len
        self.linear_x = nn.Linear(seq_len, pred_len)
        self.fusion = TrendSeasonFusion(pred_len)

    def forward(self, x_enc, components, x_dec=None, x_mark_dec=None, mask=None):
        if not isinstance(components, (tuple, list)) or len(components) < 2:
            raise ValueError('Expected trend and season references')
        trend, season = components[:2]
        offset = x_enc[:, -1:, :].detach()

        direct = self.linear_x((x_enc - offset).transpose(1, 2))
        pred = self.fusion(direct, trend.transpose(1, 2), season.transpose(1, 2)).transpose(1, 2)
        return pred + offset, 0
