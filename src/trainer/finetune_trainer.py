import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, device, model_name, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.args = args
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
            
    def _forward_epoch(self, model, batched_data):
        (smiles, g, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        predictions = model.forward_tune(g, ecfp, md)
        return predictions, labels

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        avg_train_loss = 0
        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean)/self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            avg_train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()

        
        return avg_train_loss / len(train_loader)

    def fit(self, model, train_loader, val_loader, test_loader):
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0

        for epoch in tqdm(range(1, self.args.n_epochs+1)):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            train_loss = self.train_epoch(model, train_loader, epoch)

            if self.local_rank == 0:
                val_loss_epoch, val_result = self.eval(model, val_loader)
                test_loss_epoch, test_result = self.eval(model, test_loader)
                _, train_result = self.eval(model, train_loader)
                
                if self.args.metric == 'rocauc':
                    wandb.log({'train_loss':train_loss, 'val_loss':val_loss_epoch, 'test_loss':test_loss_epoch, 'train_auc':train_result, 'val_auc':val_result, 'test_auc':test_result})
                
                elif self.args.metric == 'rmse':
                    wandb.log({'train_loss':train_loss, 'val_loss':val_loss_epoch, 'test_loss':test_loss_epoch, 'train_rmse':train_result, 'val_rmse':val_result, 'test_rmse':test_result})
                
                if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result
                    best_epoch = epoch
                if epoch - best_epoch >= 20:
                    break

        return best_train_result, best_val_result, best_test_result
    
    def eval(self, model, dataloader):
        model.eval()
        predictions_all = []
        labels_all = []

        for batched_data in dataloader:
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        loss = self.loss_fn(torch.cat(predictions_all),torch.cat(labels_all))
        return np.nanmean(np.array(loss)),result