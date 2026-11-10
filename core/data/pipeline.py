   
                        

                                                                        
   

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from core.config import BaseConfig
from .transforms import Transform


class DataPipeline(ABC):
    pass                                                      
       
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.transforms: List[Transform] = []
    
    def add_transform(self, transform: Transform):
        pass                                  
        self.transforms.append(transform)
    
    def apply_transforms(self, data: Any) -> Any:
        pass                                   
        for transform in self.transforms:
            data = transform(data)
        return data
    
    @abstractmethod
    def load_data(self, flag: str) -> Tuple[Dataset, DataLoader]:
        pass                                   
        pass


class TimeSeriesPipeline(DataPipeline):
    pass                                             
       
    
    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self._setup_default_transforms()
    
    def _setup_default_transforms(self):
        pass                                                
                                           
        pass
    
    def load_data(self, flag: str) -> Tuple[Dataset, DataLoader]:
        pass                                                     
        
             
                                                          
            
                
                                          
           
                                                       
        from data_provider.data_factory import data_provider as legacy_data_provider
        
        dataset, dataloader = legacy_data_provider(self.config, flag)
        
                                 
        if self.transforms:
                                                              
            dataset = TransformDataset(dataset, self.transforms)
            
                                                          
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(flag == 'train'),
                num_workers=self.config.num_workers,
                drop_last=False,
                pin_memory=True
            )
        
        return dataset, dataloader


class TransformDataset(Dataset):
    pass                                                    
       
    
    def __init__(self, dataset: Dataset, transforms: List[Transform]):
        self.dataset = dataset
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
                          
        for transform in self.transforms:
            data = transform(data)
        
        return data


class HighDimensionalPipeline(TimeSeriesPipeline):
    pass                                                               
       
    
    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self._setup_high_dimensional_transforms()
    
    def _setup_high_dimensional_transforms(self):
        pass                                                     
                                                              
        pass 