import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.CCM_layers import *
    
class moving_avg(nn.Module):
    pass                                                              
       
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
                                                 
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    pass                              
       
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    pass                        
       
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_cluster = args.n_cluster
        self.d_ff = args.d_ff

                                  
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = args.enc_in

        self.Linear_Seasonal = Cluster_wise_linear(self.n_cluster, self.channels, self.seq_len, self.pred_len, device=args.accelerator.device)
        self.Linear_Trend = Cluster_wise_linear(self.n_cluster, self.channels,self.seq_len, self.pred_len, device=args.accelerator.device)
        self.Cluster_assigner = Cluster_assigner(self.channels, self.n_cluster, self.seq_len, self.d_ff, device=args.accelerator.device)
                                                                            
                                                                      
        self.register_buffer('cluster_emb', self.Cluster_assigner.cluster_emb.clone())
            

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        
        batch_x = x

        self.cluster_prob, cluster_emb = self.Cluster_assigner(x, self.cluster_emb.clone())

        if self.training:
            self.cluster_emb.copy_(cluster_emb.detach())
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        seasonal_output = self.Linear_Seasonal(seasonal_init, self.cluster_prob)
        trend_output = self.Linear_Trend(trend_init, self.cluster_prob)

        x = seasonal_output + trend_output

        loss_aux = self.similarity_loss_batch(batch_x)
        
        return x.permute(0,2,1), loss_aux
    
    def get_similarity_matrix_update(self, batch_data, sigma=None, use_median=True):
        pass                                                                                             
                                                                                                   
                                                      
        
                   
                                                                                         
                                                                                                
            
                
                                                                        
           
                                                             
                                                                                                 

                                                                        
                                                                                                             
        
                                                                            
                                                                                    
        
                                       
                                       
                                           
                                                                                                 
                                                                               

                                                        
                                                

        time_diffs = batch_data[:, 1:, :] - batch_data[:, :-1, :]

                           
        channel_repr = time_diffs.mean(dim=(0, 1))
        channel_repr = (channel_repr - channel_repr.mean()) / (channel_repr.std() + 1e-6)

                                                                 
        diff_matrix = channel_repr.unsqueeze(0) - channel_repr.unsqueeze(1)                             
        squared_diff_matrix = diff_matrix ** 2

        if sigma is None:
                                           
            c = squared_diff_matrix.shape[0]
            mask = ~torch.eye(c, dtype=torch.bool, device=batch_data.device)
            off_diag_values = squared_diff_matrix[mask]
            
            if use_median:
                sigma = torch.sqrt(torch.median(off_diag_values))
            else:
                sigma = torch.sqrt(torch.mean(off_diag_values))

                             
            sigma = sigma.clamp(min=1e-5)

                                   
        similarity_matrix = torch.exp(-squared_diff_matrix / (2 * sigma ** 2))

        return similarity_matrix.to(batch_data.device)
    
                                                  
                                               
                                                                                                    
                                                                                    
                                                                            
                                                                  
                              
                                                                   
                                                                              
                                                       
                                            
                                                        
                                                                                                
                                                                                                   
                                
    def similarity_loss_batch(self, batch_data):
        def concrete_bern(prob, temp=0.07):
                                                                   
            eps = 1e-10
            noise = torch.empty_like(prob).uniform_(eps, 1 - eps)
            noise = torch.log(noise) - torch.log(1.0 - noise)
            logit = torch.log(prob + eps) - torch.log(1.0 - prob + eps)
            return ((logit + noise) / temp).sigmoid()

        sim_matrix = self.get_similarity_matrix_update(batch_data)                    
        membership = concrete_bern(self.cluster_prob)                        

                                                             
        sim_proj = membership.T @ sim_matrix @ membership                            
                                      
        mem_dot = membership @ membership.T                    
        orth_loss = 1 - mem_dot

                                                                                                    
                                                                                                   
        
        n_vars = membership.shape[0]

                                                                
        entropy = (-self.cluster_prob * torch.log(self.cluster_prob + 1e-15)).sum(dim=-1).mean()

        loss = (
            -torch.trace(sim_proj) / n_vars +
            torch.trace(orth_loss @ sim_matrix) / (n_vars ** 2) +
            entropy
        )

        return loss + entropy
