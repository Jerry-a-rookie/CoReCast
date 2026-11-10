   
                       

                                                                            
                                                                         
                                      
   

from typing import Tuple, Any, Optional
from torch.utils.data import Dataset, DataLoader
from core.config import BaseConfig
from .data_factory import data_provider as core_data_provider


def data_provider(config: BaseConfig, flag: str, accelerator=None) -> Tuple[Dataset, DataLoader]:
    pass                                                             
    
                                                                        
                                                                     
                                                                 
    
         
                                                                                    
                                                                    
                                                                              
        
            
                         
                                                                      
                                                                    
       
                                               
    return core_data_provider(config, flag, accelerator) 