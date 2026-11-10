   
                 

                                                          
   

import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricsCalculator:
    pass                                                   
       
    
    @staticmethod
    def regression_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        pass                                     
        
             
                                  
                             
            
                
                                 
           
        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((pred - true) / true))
        mspe = np.mean(((pred - true) / true) ** 2)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe
        }
    
    @staticmethod
    def classification_metrics(pred: np.ndarray, true: np.ndarray, 
                             average: str = 'macro') -> Dict[str, float]:
        pass                                         
        
             
                                        
                                   
                                                             
            
                
                                 
           
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred, average=average, zero_division=0)
        recall = recall_score(true, pred, average=average, zero_division=0)
        f1 = f1_score(true, pred, average=average, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def anomaly_detection_metrics(pred: np.ndarray, true: np.ndarray, 
                                threshold: float = 0.5) -> Dict[str, float]:
        pass                                            
        
             
                                          
                                                                  
                                                                            
            
                
                                 
           
                                              
        pred_binary = (pred > threshold).astype(int)
        
                                                
        metrics = MetricsCalculator.classification_metrics(pred_binary, true)
        
                                      
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            auc = roc_auc_score(true, pred)
            ap = average_precision_score(true, pred)
            
            metrics.update({
                'auc': auc,
                'average_precision': ap
            })
        except ImportError:
            pass
        
        return metrics
    
    @staticmethod
    def imputation_metrics(pred: np.ndarray, true: np.ndarray, 
                          mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        pass                                     
        
             
                                            
                             
                                                                                
            
                
                                 
           
        if mask is not None:
                                                      
            pred_masked = pred[mask == 1]
            true_masked = true[mask == 1]
        else:
            pred_masked = pred
            true_masked = true
        
        return MetricsCalculator.regression_metrics(pred_masked, true_masked)


class EarlyStopping:
    pass                                                                 
       
    
    def __init__(self, patience: int = 7, verbose: bool = False, 
                 delta: float = 0, path: str = 'checkpoint.pt',
                 trace_func=print):
        pass             
                                                                               
                                                                                   
                                                                                        
                                                        
                                            
           
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        pass                                            
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/checkpoint.pth')
        self.val_loss_min = val_loss


class ProgressTracker:
    pass                                                       
       
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        self.best_metrics = {}
    
    def update(self, epoch: int, **metrics):
        pass                                   
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        pass                                   
        if metric not in self.history:
            return -1
        
        values = self.history[metric]
        if mode == 'min':
            return np.argmin(values)
        else:
            return np.argmax(values)
    
    def save_history(self, path: str):
        pass                                
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.history, f)
    
    def load_history(self, path: str):
        pass                                  
        import pickle
        with open(path, 'rb') as f:
            self.history = pickle.load(f)


def adjust_learning_rate(optimizer, epoch: int, args):
    pass                                               
    
         
                                    
                            
                                                                    
       
    if hasattr(args, 'lradj') and args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif hasattr(args, 'lradj') and args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        lr_adjust = {}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


def visual(true, preds=None, name='./pic/test.pdf'):
    pass                         
       
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe 