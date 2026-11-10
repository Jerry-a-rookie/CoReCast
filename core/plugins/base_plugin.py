   
                  

                                                          
   

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List


class BasePlugin(ABC):
    pass                               
       
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = True
        self.dependencies = []
    
    @abstractmethod
    def initialize(self, **kwargs):
        pass                                           
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass                                          
        pass
    
    def cleanup(self):
        pass                                             
        pass
    
    def get_info(self) -> Dict[str, Any]:
        pass                         
        return {
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled,
            'dependencies': self.dependencies
        }


class ModelPlugin(BasePlugin):
    pass                                 
       
    
    def __init__(self, name: str, model_class, version: str = "1.0.0"):
        super().__init__(name, version)
        self.model_class = model_class
    
    @abstractmethod
    def create_model(self, config: Any) -> Any:
        pass                                                 
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        pass                                 
        info = self.get_info()
        info.update({
            'model_class': self.model_class.__name__ if self.model_class else None,
            'type': 'model'
        })
        return info


class DataPlugin(BasePlugin):
    pass                                           
       
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
    
    @abstractmethod
    def process_data(self, data: Any, **kwargs) -> Any:
        pass                                         
        pass
    
    def get_data_info(self) -> Dict[str, Any]:
        pass                                
        info = self.get_info()
        info['type'] = 'data'
        return info


class MetricsPlugin(BasePlugin):
    pass                                   
       
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
    
    @abstractmethod
    def calculate_metrics(self, predictions: Any, targets: Any, **kwargs) -> Dict[str, float]:
        pass                                                
        pass
    
    def get_metrics_info(self) -> Dict[str, Any]:
        pass                                   
        info = self.get_info()
        info['type'] = 'metrics'
        return info


class VisualizationPlugin(BasePlugin):
    pass                                         
       
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
    
    @abstractmethod
    def create_visualization(self, data: Any, **kwargs) -> Any:
        pass                                 
        pass
    
    def get_visualization_info(self) -> Dict[str, Any]:
        pass                                         
        info = self.get_info()
        info['type'] = 'visualization'
        return info


class HookPlugin(BasePlugin):
    pass                                                                             
       
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self.hook_points = []
    
    def on_experiment_start(self, experiment, **kwargs):
        pass                                
        pass
    
    def on_experiment_end(self, experiment, results, **kwargs):
        pass                              
        pass
    
    def on_epoch_start(self, experiment, epoch, **kwargs):
        pass                                    
        pass
    
    def on_epoch_end(self, experiment, epoch, metrics, **kwargs):
        pass                                  
        pass
    
    def on_batch_start(self, experiment, batch_idx, **kwargs):
        pass                                    
        pass
    
    def on_batch_end(self, experiment, batch_idx, loss, **kwargs):
        pass                                  
        pass
    
    def get_hook_info(self) -> Dict[str, Any]:
        pass                                
        info = self.get_info()
        info.update({
            'type': 'hook',
            'hook_points': self.hook_points
        })
        return info 