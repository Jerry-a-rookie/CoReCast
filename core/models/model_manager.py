   
             

                                                                             
                                                              
   

import torch.nn as nn
from typing import Optional, Dict, Any
from core.registry import model_registry
from core.config import BaseConfig


class ModelManager:
    pass                                          
    
                                                                                             
                                                                                        
       
    
    def __init__(self):
        pass                                                              
        self.registry = model_registry
        self._ensure_models_loaded()
    
    def _ensure_models_loaded(self):
        pass                                                              
        
                                                                              
                                                          
           
        try:
                                                   
            import models.UCast                                      
            import models.ComponentRAFT
            import models.ComponentRAFTSeasonTrend
            import models.DLinear
            import models.TimesNet
            
                                                         
            import models.Autoformer
            import models.Informer
            import models.FEDformer
            import models.PatchTST
            import models.iTransformer
            import models.Transformer
            import models.Nonstationary_Transformer
            import models.ETSformer
            import models.Crossformer
            import models.Pyraformer
            
                                     
            import models.ModernTCN
            import models.MICN
            
                                                     
            import models.TiDE
            import models.SegRNN
            import models.LightTS
            
                                                          
            import models.TSMixer
            import models.FreTS
            
                                                    
            
        except ImportError as e:
            print(f"Warning: Failed to import some models - they may not be available: {e}")
    
    def create_model(self, model_name: str, config: BaseConfig) -> nn.Module:
        pass                                               
        
             
                                                              
                                                                    
            
                
                                              
            
               
                                                                                  
           
        return self.registry.create_model(model_name, config)
    
    def get_model_class(self, model_name: str) -> Optional[type]:
        pass                                                             
        
             
                                                    
            
                
                                            
           
        return self.registry.get_model(model_name)
    
    def list_available_models(self) -> list:
        pass                                                       
        
                
                                                                
           
        return sorted(self.registry.list_models())
    
    def is_model_available(self, model_name: str) -> bool:
        pass                                                      
        
             
                                                  
            
                
                                                        
           
        return model_name in self.registry
    
    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        pass                                                 
        
                                                                  
                                                            
        
             
                                                    
            
                
                                                
           
        return self.registry.get_metadata(model_name)
    
    def register_model(self, name: str, model_class: type, **metadata) -> type:
        pass                                              
        
                                                                    
                                                             
        
             
                                                         
                                                        
                                                                    
            
                
                                      
            
               
                                                                    
           
        return self.registry.register_model(name, model_class, **metadata)
    
    def get_model_count(self) -> int:
        pass                                                  
        
                
                                                 
           
        return len(self.registry)
    
    def print_model_summary(self):
        pass                                                               
        models = self.list_available_models()
        print(f"\n=== High-Dimensional Time Series Framework - Registered Models ({len(models)}) ===")
        
        for model_name in models:
            metadata = self.get_model_metadata(model_name)
            year = metadata.get('year', 'Unknown')
            paper = metadata.get('paper', 'No paper information')
            print(f"  • {model_name:<20} ({year}) - {paper}")
        
        print(f"\nTotal: {len(models)} models available")


                               
model_manager = ModelManager() 
