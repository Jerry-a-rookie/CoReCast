   
                         

                                                                  
   

import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class BaseConfig:
    pass                                                                
    
                                                                  
       
    
    def __post_init__(self):
        pass                                          
        self.validate()
    
    def validate(self):
        pass                                                            
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        pass                                      
        return asdict(self)
    
    def to_yaml(self, file_path: Optional[Union[str, Path]] = None) -> str:
        pass                                             
        
             
                                                          
            
                
                                      
           
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        
        if file_path is not None:
            with open(file_path, 'w') as f:
                f.write(yaml_str)
        
        return yaml_str
    
    def to_json(self, file_path: Optional[Union[str, Path]] = None) -> str:
        pass                                             
        
             
                                                          
            
                
                                      
           
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if file_path is not None:
            with open(file_path, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        pass                                             
        
             
                                                                       
            
                
                                  
           
                              
        import inspect
        from dataclasses import fields
        
                                                 
        if hasattr(cls, '__dataclass_fields__'):
            valid_keys = set(cls.__dataclass_fields__.keys())
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
            
                                               
            instance = cls(**filtered_dict)
            
                                                     
            for k, v in config_dict.items():
                if k not in valid_keys:
                    setattr(instance, k, v)
        else:
                                        
            sig = inspect.signature(cls)
            valid_keys = set(sig.parameters.keys())
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
            instance = cls(**filtered_dict)
            
                                                     
            for k, v in config_dict.items():
                if k not in valid_keys:
                    setattr(instance, k, v)
        
        return instance
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'BaseConfig':
        pass                                            
        
             
                                        
            
                
                                  
           
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod 
    def from_json(cls, file_path: Union[str, Path]) -> 'BaseConfig':
        pass                                            
        
             
                                        
            
                
                                  
           
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs) -> 'BaseConfig':
        pass                                             
        
             
                                              
            
                
                                                          
           
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__.from_dict(config_dict)
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        pass                                                            
        
             
                                                      
            
                
                                                         
           
        config_dict = self.to_dict()
        other_dict = other.to_dict()
        config_dict.update(other_dict)
        return self.__class__.from_dict(config_dict)
    
    def __repr__(self) -> str:
        pass                                             
        return f"{self.__class__.__name__}({self.to_dict()})" 