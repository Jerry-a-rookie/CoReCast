   
                     

                                                                                                    
   

import os
import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from core.config import BaseConfig
from core.models import model_manager
from core.data import data_provider
from core.registry import model_registry


class BaseExperiment(ABC):
    pass                                                                        
    
                                                                                          
       
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.accelerator = getattr(config, 'accelerator', None)
        self.device = self.accelerator.device if self.accelerator else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device_info_printed = False                                                 
        
                                  
        if self.accelerator and torch.cuda.is_available():
            self._print_device_info()
        
                                                
        self.model = self._build_model()
    
    def _print_device_info(self):
        pass                                     
        if self.accelerator and not self._device_info_printed:
            self.accelerator.print("\n=== Accelerator Device Information ===")
            self.accelerator.print(f"Number of processes: {self.accelerator.num_processes}")
            self.accelerator.print(f"Distributed type: {self.accelerator.distributed_type}")
            self.accelerator.print("\nDevice details for all processes:")
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                self.accelerator.print(f"  GPU #{i}: {device_props.name} - Total memory: {device_props.total_memory / 1024**3:.2f} GB")
            self.accelerator.print("=======================================\n")
            self._device_info_printed = True
    
    def _build_model(self) -> torch.nn.Module:
        pass                                                
        
                
                             
           
        model_name = getattr(self.config, 'model', 'Transformer')
        
                                                     
        if model_manager.is_model_available(model_name):
            model = model_manager.create_model(model_name, self.config)
        else:
                                                                
            model = self._build_model_legacy(model_name)
        
        return model
    
    def _build_model_legacy(self, model_name: str) -> torch.nn.Module:
        pass                                                             
                                                          
           
                                                     
        from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
            Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
            TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, MultiPatchFormer,\
            ModernTCN, CCM, PDF, DUET
        
        model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'MultiPatchFormer': MultiPatchFormer,
            'ModernTCN': ModernTCN,
            'CCM': CCM,
            'PDF': PDF,
            'DUET': DUET,
        }
        
        if model_name == 'Mamba':
            from models import Mamba
            model_dict['Mamba'] = Mamba
        
        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} not found in legacy model dictionary")
        
        return model_dict[model_name].Model(self.config)
    
    @abstractmethod
    def _get_data(self, flag: str):
        pass                                                         
        
             
                                 
            
                
                                  
           
        pass
    
    @abstractmethod
    def train(self, setting: str):
        pass                        
        
             
                                              
           
        pass
    
    @abstractmethod
    def test(self, setting: str):
        pass                       
        
             
                                              
           
        pass
    
    @abstractmethod
    def validate(self):
        pass                     
        pass
    
    def set_accelerator(self, accelerator):
        pass                                         
        self.accelerator = accelerator
        if accelerator:
            self.device = accelerator.device
                                                                  
            if not self._device_info_printed:
                self._print_device_info() 