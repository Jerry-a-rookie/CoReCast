   
                      

                                                                       
   

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Transform(ABC):
    pass                                            
       
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        pass                                             
        
             
                                         
            
                
                            
           
        pass
    
    def set_params(self, **kwargs):
        pass                                   
        self.params.update(kwargs)
    
    def get_params(self) -> Dict[str, Any]:
        pass                                
        return self.params.copy()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"


class CompositeTransform(Transform):
    pass                                                                          
       
    
    def __init__(self, transforms: list, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms
    
    def __call__(self, data: Any) -> Any:
        pass                                   
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def add_transform(self, transform: Transform):
        pass                                  
        self.transforms.append(transform)
    
    def __repr__(self) -> str:
        transform_names = [transform.__class__.__name__ for transform in self.transforms]
        return f"CompositeTransform([{', '.join(transform_names)}])" 