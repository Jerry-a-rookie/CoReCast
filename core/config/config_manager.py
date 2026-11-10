   
                     

                                                                                   
   

import os
import yaml
import argparse
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .base_config import BaseConfig
from .model_configs import ModelConfig, get_model_config_class


class ConfigManager:
    pass                                      
    
                                                                                     
       
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._loaded_files: set = set()
        
                                      
        self._load_all_configs()
    
    def _load_all_configs(self):
        pass                                                         
        if not self.config_dir.exists():
            print(f"Warning: Config directory {self.config_dir} does not exist")
            return
        
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                self._load_config_file(config_file)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_file}: {e}")
    
    def _load_config_file(self, file_path: Path):
        pass                                   
        if file_path in self._loaded_files:
            return
        
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if config_data is None:
            return
        
                                            
        model_name = file_path.stem
        self._model_configs[model_name] = config_data
        self._loaded_files.add(file_path)
    
    def get_model_config(self, model_name: str, dataset_name: str, **overrides) -> ModelConfig:
        pass                                           
        
             
                                         
                                                                                  
                                                           
            
                
                                
           
                                          
        config_class = get_model_config_class(model_name)
        
                                          
        config_dict = {}
        
                                                     
        if model_name in self._model_configs:
            model_file_config = self._model_configs[model_name]
            
                                                      
            if dataset_name in model_file_config:
                config_dict.update(model_file_config[dataset_name])
            else:
                                                 
                for key, value in model_file_config.items():
                    if not isinstance(value, dict):
                        config_dict[key] = value
        
                         
        config_dict.update(overrides)
        
                                           
        return config_class.from_dict(config_dict)
    
    def create_unified_config(self, model_name: str, dataset_name: str, **overrides) -> BaseConfig:
        pass                                                     
        
             
                                         
                                             
                                                           
            
                
                                        
           
                          
        model_config = self.get_model_config(model_name, dataset_name)
        
                              
        unified_dict = model_config.to_dict()
        unified_dict.update(overrides)
        
                                                                  
        config_class = get_model_config_class(model_name)
        return config_class.from_dict(unified_dict)
    
    def parse_args_and_create_config(self, args: argparse.Namespace) -> BaseConfig:
        pass                                                         
        
             
                                               
            
                
                                        
           
                                                                       
        args_dict = {k: v for k, v in vars(args).items() if v is not None}
        
                                         
        model_name = args_dict.get('model', 'DLinear')
        dataset_name = args_dict.get('data', 'SIRS')
        
                                      
        config = self.create_unified_config(model_name, dataset_name, **args_dict)
        
        return config
    
    def list_available_models(self) -> list:
        pass                                                   
        return list(self._model_configs.keys())
    
    def reload_configs(self):
        pass                                 
        self._model_configs.clear()
        self._loaded_files.clear()
        self._load_all_configs()
    
    def add_config_source(self, source_path: Union[str, Path]):
        pass                                         
        source_path = Path(source_path)
        
        if source_path.is_file():
            self._load_config_file(source_path)
        elif source_path.is_dir():
            for config_file in source_path.glob("*.yaml"):
                self._load_config_file(config_file)
        else:
            raise ValueError(f"Invalid config source: {source_path}")


                                       
config_manager = ConfigManager() 