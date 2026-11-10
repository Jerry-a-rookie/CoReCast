   
                 

                                                                          
                                                          
   

import os
import torch
from typing import Dict, Any, Tuple
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from core.config import BaseConfig, ConfigManager
from core.models import ModelManager
from core.experiments.long_term_forecasting import LongTermForecastingExperiment
                                                                                    
                                                                      
                                                              
                                                                           


class ExperimentRunner:
    pass                                                      
    
                                                                           
                                                        
       
    
    def __init__(self, config: BaseConfig):
        pass                                         
        
             
                                                                         
           
        self.config = config
        self.accelerator = None
        self._setup_environment()
    
    def _setup_environment(self):
        pass                                                                            
                                              
        random_seed = getattr(self.config, 'seed', 2021)
        self._fix_seed(random_seed)
        
                                                             
        try:
            kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(kwargs_handlers=[kwargs])
        except Exception as e:
            print(f"Warning: Could not initialize accelerator: {e}")
            self.accelerator = None
        
                            
        use_gpu = getattr(self.config, 'use_gpu', True)

        if torch.cuda.is_available() and use_gpu:
            self.device = self.accelerator.device if self.accelerator else torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
    
    def _fix_seed(self, seed: int):
        pass                                                      
        
             
                                   
           
        import random
        import numpy as np
        
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _create_experiment(self) -> Any:
        pass                                                                      
        
                
                                                      
            
               
                                                         
           
        experiment_map = {
            'long_term_forecast': LongTermForecastingExperiment,
                                                           
                                                                    
                                                         
                                                 
                                                              
        }
        
        experiment_class = experiment_map.get(self.config.task_name)
        if experiment_class is None:
            available_tasks = ", ".join(experiment_map.keys())
            raise ValueError(f"Unsupported task: '{self.config.task_name}'. Available tasks: [{available_tasks}]")
        
        return experiment_class(self.config)
    
    def run(self) -> Dict[str, Any]:
        pass                                                 
        
                
                                                                                
                                                        
           
                                                                    
        setting = self._create_setting_string()
        
                                                  
        experiment = self._create_experiment()
        experiment.set_accelerator(self.accelerator)
        
        results = {}
        
        if self.config.is_training:
            self.accelerator.print(f'>>> Starting training for {setting} <<<')
            
                                    
            all_epoch_metrics, best_metrics, best_model_path = experiment.train(setting)
            
            results['training'] = {
                'all_epoch_metrics': all_epoch_metrics,
                'best_metrics': best_metrics,
                'best_model_path': best_model_path
            }
            
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.ipc_collect()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                
                                              
            self.accelerator.print(f'>>> Starting testing for {setting} <<<')
            mse, mae = experiment.test(setting, best_model_path)
            
            results['testing'] = {
                'mse': mse,
                'mae': mae
            }
            
        else:
                               
            self.accelerator.print(f'>>> Starting testing for {setting} <<<')
            mse, mae = experiment.test(setting)
            
            results['testing'] = {
                'mse': mse,
                'mae': mae
            }
            
            self.accelerator.print(f'>>> Testing completed for {setting} <<<')
        
        return results
    
    def _create_setting_string(self) -> str:
        pass                                                                     
        
                
                                                                     
           
                                                                     
        task = getattr(self.config, 'task_name', 'forecast')
        model = getattr(self.config, 'model', 'UCast')
        data = getattr(self.config, 'data', 'data').replace(' ', '_')                                  
        seq_len = getattr(self.config, 'seq_len', 96)
        pred_len = getattr(self.config, 'pred_len', 96)
        
                                                        
        setting = f"{task}_{model}_{data}_sl{seq_len}_pl{pred_len}"
        
        return setting


class BatchRunner:
    pass                                                            
    
                                                                  
                                                          
       
    
    def __init__(self, config_manager: ConfigManager):
        pass                                    
        
             
                                                                                 
           
        self.config_manager = config_manager
        self.experiments = []
    
    def add_experiment(self, **config_updates):
        pass                                                           
        
             
                                                                                      
           
        config = self.config_manager.get_unified_config()
        config = config.update(**config_updates)
        self.experiments.append(config)
    
    def run_batch(self) -> list:
        pass                                                   
        
                
                                                                                         
           
        batch_results = []
        
        for i, config in enumerate(self.experiments):
            print(f'\n=== Running Experiment {i+1}/{len(self.experiments)} ===')
            print(f'Configuration: Model={config.model}, Data={config.data}, Pred_len={config.pred_len}')
            
                                       
            runner = ExperimentRunner(config)
            experiment_result = runner.run()
            
            batch_results.append({
                'experiment_id': i + 1,
                'config': config.to_dict(),
                'results': experiment_result
            })
            
            print(f'Experiment {i+1} completed successfully')
        
        return batch_results


def create_runner_from_args(args) -> ExperimentRunner:
    pass                                                            
    
         
                                                         
        
            
                                                                
       
                                  
    config_manager = ConfigManager()
    
                                                
    config = config_manager.parse_args_and_create_config(args)
    
                                         
    return ExperimentRunner(config) 