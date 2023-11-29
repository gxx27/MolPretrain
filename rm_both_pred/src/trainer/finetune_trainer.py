import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, model_name, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.args = args
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
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
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx-1)*len(train_loader)+batch_idx+1)
        
        return avg_train_loss/len(train_loader)
    
    def plot_figure(self, train_loss, val_loss, test_loss, train_metric, val_metric, test_metric):
            task_name = self.args.dataset
            plt.figure()
            plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training Loss on task {task_name}')
            plt.legend()
            plt.savefig(f'train_loss_{task_name}.png')

            plt.figure()
            plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Validation Loss on task {task_name}')
            plt.legend()
            plt.savefig(f'val_loss_{task_name}.png')      

            plt.figure()
            plt.plot(range(len(test_loss)), test_loss, label='Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Test Loss on task {task_name}')
            plt.legend()
            plt.savefig(f'test_loss_{task_name}.png')

            if self.args.metric == 'rocauc':
                plt.figure()
                plt.plot(range(len(train_metric)), train_metric, label='Training auc')
                plt.xlabel('Epochs')
                plt.ylabel('Auc')
                plt.title(f'Training auc on task {task_name}')
                plt.legend()
                plt.savefig(f'train_auc_{task_name}.png')

                plt.figure()
                plt.plot(range(len(val_metric)), val_metric, label='Validation auc')
                plt.xlabel('Epochs')
                plt.ylabel('Auc')
                plt.title(f'Validation auc on task {task_name}')
                plt.legend()
                plt.savefig(f'val_auc_{task_name}.png')      

                plt.figure()
                plt.plot(range(len(test_metric)), test_metric, label='Test auc')
                plt.xlabel('Epochs')
                plt.ylabel('Auc')
                plt.title(f'Test auc on task {task_name}')
                plt.legend()
                plt.savefig(f'test_auc_{task_name}.png')

            else:
                plt.figure()
                plt.plot(range(len(train_metric)), train_metric, label='Training rmse')
                plt.xlabel('Epochs')
                plt.ylabel('RMSE')
                plt.title(f'Training rmse on task {task_name}')
                plt.legend()
                plt.savefig(f'train_rmse_{task_name}.png')

                plt.figure()
                plt.plot(range(len(val_metric)), val_metric, label='Validation rmse')
                plt.xlabel('Epochs')
                plt.ylabel('RMSE')
                plt.title(f'Validation rmse on task {task_name}')
                plt.legend()
                plt.savefig(f'val_rmse_{task_name}.png')      

                plt.figure()
                plt.plot(range(len(test_metric)), test_metric, label='Test rmse')
                plt.xlabel('Epochs')
                plt.ylabel('RMSE')
                plt.title(f'Test rmse on task {task_name}')
                plt.legend()
                plt.savefig(f'test_rmse_{task_name}.png')

    def fit(self, model, train_loader, val_loader, test_loader):
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0
        train_loss = []
        train_metric = []
        val_loss = []
        val_metric = []
        test_loss = []
        test_metric = []
        for epoch in tqdm(range(1, self.args.n_epochs+1)):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            train_loss.append(self.train_epoch(model, train_loader, epoch))
            # self.train_epoch(model, train_loader, epoch)
            if self.local_rank == 0:
                val_loss_epoch, val_result = self.eval(model, val_loader)
                test_loss_epoch, test_result = self.eval(model, test_loader)
                _, train_result = self.eval(model, train_loader)
                train_metric.append(train_result)
                val_loss.append(val_loss_epoch)
                val_metric.append(val_result)
                test_loss.append(test_loss_epoch)
                test_metric.append(test_result)
                if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result
                    best_epoch = epoch
                if epoch - best_epoch >= 20:
                    break
        self.plot_figure(train_loss,val_loss,test_loss,train_metric,val_metric,test_metric)
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

    