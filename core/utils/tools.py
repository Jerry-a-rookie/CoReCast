import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv
import json

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, accelerator=None):
                                                     
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'TSLR':
        lr_adjust = {epoch: args.learning_rate * ((0.5 ** 0.1) ** (epoch // 20))}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: args.learning_rate * (1.0 + 0.1 * epoch / args.train_epochs)}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if accelerator is not None:
            accelerator.print('Updating learning rate to {}'.format(lr))
        else:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, accelerator=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.accelerator = accelerator
        self.best_metrics = None                                 

    def __call__(self, val_loss, model, path, metrics=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, metrics)
            if metrics is not None:
                self.best_metrics = metrics
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, metrics)
            if metrics is not None:
                self.best_metrics = metrics
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, metrics=None):
        if self.verbose:
                                                                      
            if self.accelerator:
                self.accelerator.print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
                                             
        checkpoints_dir = './checkpoints'
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir, exist_ok=True)
        
                                                                 
        experiment_name = os.path.basename(path)
        checkpoint_file = os.path.join(checkpoints_dir, f'{experiment_name}.pth')

        if self.accelerator:
            model = self.accelerator.unwrap_model(model)
        
        torch.save(model.state_dict(), checkpoint_file)
        
        self.val_loss_min = val_loss
        
                                                 
        self.checkpoint_file = checkpoint_file
        
                                                            
        if metrics is not None:
            metrics_file = os.path.join(checkpoints_dir, f'{experiment_name}_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)

    def get_best_metrics(self):
        pass                                    
        return self.best_metrics
    
    def get_checkpoint_path(self):
        return getattr(self, "checkpoint_file", None)


class dotdict(dict):
    pass                                              
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    pass                         
       
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def save_results_to_csv(args, best_result, hp_results_dir=None):
    pass                                                   
    
         
                                                                       
                                                                                                                
                                                                                       
    
            
                                                 
       
                                        
    csv_file_path = os.path.join(args.hp_log_dir, 'results.csv')
    
                           
    model_method = f"{args.model}"              
    current_dataset = args.data                            
    
                              
    csv_exists = os.path.exists(csv_file_path)
    
                                                   
    if not csv_exists:
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset', 'pred_len', f'{model_method}_MSE', f'{model_method}_MAE'])
        
                                              
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for result in best_result['pred_len_results']:
                pred_len = result['pred_len']
                test_loss = result['test_loss']
                test_mae = result['test_mae_loss']
                writer.writerow([current_dataset, pred_len, test_loss, test_mae])
                
                           
        if hp_results_dir:
            args.accelerator.print(f"Results created in: {csv_file_path}")
        
        return csv_file_path
    
                                                     
    rows_to_write = []
    existing_rows = []
    
                        
    try:
        with open(csv_file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            existing_rows = list(reader)
            
                                                        
            if len(existing_rows) == 0:
                if hp_results_dir:
                    args.accelerator.print(f"CSV file exists but is empty: {csv_file_path}")
                                                
                existing_rows = [['dataset', 'pred_len', f'{model_method}_MSE', f'{model_method}_MAE']]
                                          
                with open(csv_file_path, 'w', newline='') as empty_file:
                    writer = csv.writer(empty_file)
                    writer.writerow(existing_rows[0])
                    
    except Exception as e:
        if hp_results_dir:
            args.accelerator.print(f"Error reading CSV file: {e}, creating new file")
                                           
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset', 'pred_len', f'{model_method}_MSE', f'{model_method}_MAE'])
        existing_rows = [['dataset', 'pred_len', f'{model_method}_MSE', f'{model_method}_MAE']]
    
                                                    
    if not existing_rows:
        existing_rows = [['dataset', 'pred_len', f'{model_method}_MSE', f'{model_method}_MAE']]
    
    header = existing_rows[0]
    
                                           
    mse_col = f'{model_method}_MSE'
    mae_col = f'{model_method}_MAE'
    
    if mse_col not in header:
        header.append(mse_col)
    if mae_col not in header:
        header.append(mae_col)
    
                        
    mse_index = header.index(mse_col)
    mae_index = header.index(mae_col)
    
                                         
    rows_to_write.append(header)
    
                                         
    for row in existing_rows[1:]:
                                           
        while len(row) < len(header):
            row.append('')
        rows_to_write.append(row)
    
                                    
    for result in best_result['pred_len_results']:
        pred_len = result['pred_len']
        test_loss = result['test_loss']
        test_mae = result['test_mae_loss']
        
                                                
        row_found = False
        
                                                                    
        for i in range(1, len(rows_to_write)):
            row = rows_to_write[i]
            if len(row) >= 2 and row[0] == current_dataset:
                try:
                    row_pred_len = float(row[1])
                    if int(row_pred_len) == int(pred_len):
                                         
                        row[mse_index] = str(test_loss)
                        row[mae_index] = str(test_mae)
                        rows_to_write[i] = row
                        row_found = True
                        break
                except (ValueError, IndexError):
                    continue
        
                                               
        if not row_found:
            new_row = [''] * len(header)
            new_row[0] = current_dataset
            new_row[1] = str(pred_len)
            new_row[mse_index] = str(test_loss)
            new_row[mae_index] = str(test_mae)
            rows_to_write.append(new_row)
    
                                
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows_to_write)
    
                       
    if hp_results_dir:
        args.accelerator.print(f"Results appended to: {csv_file_path}")
    
    return csv_file_path