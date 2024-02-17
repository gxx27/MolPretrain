import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from copy import deepcopy

class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, sl_loss_fn, 
                 reg_evaluator, clf_evaluator, result_tracker, summary_writer, device, ddp=False, local_rank=1):
        
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.reg_loss_fn = reg_loss_fn
        self.clf_loss_fn = clf_loss_fn
        self.sl_loss_fn = sl_loss_fn
        self.reg_evaluator = reg_evaluator
        self.clf_evaluator = clf_evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.ddp = ddp
        self.local_rank = local_rank
        self.n_updates = 0      
    
        self.training_updates = 0
        self.train_episode = 1 # mark training episode 1 for the first and 2 for the second
    
    def _forward_epoch(self, model, batched_data):
        (smiles, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds, subgraph_labels) = batched_data
        batched_graph = batched_graph.to(self.device)
        fps = fps.to(self.device)
        mds = mds.to(self.device)
        sl_labels = sl_labels.to(self.device)
        disturbed_fps = disturbed_fps.to(self.device)
        disturbed_mds = disturbed_mds.to(self.device)
        subgraph_labels = subgraph_labels.to(self.device)
        sl_predictions, fp_predictions, md_predictions, subgraph_prediction  = model(batched_graph, disturbed_fps, disturbed_mds, fps, mds)

        mask_replace_keep = batched_graph.ndata['mask'][batched_graph.ndata['mask']>=1].cpu().numpy()
        
        return mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds, subgraph_prediction, subgraph_labels
    
    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            try:
                self.optimizer.zero_grad()
                mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds, subgraph_prediction, subgraph_labels = self._forward_epoch(model, batched_data)
                sl_loss = self.sl_loss_fn(sl_predictions, sl_labels).mean()
                fp_loss = self.clf_loss_fn(fp_predictions, fps).mean()
                md_loss = self.reg_loss_fn(md_predictions, mds).mean()
                subgraph_loss = self.sl_loss_fn(subgraph_prediction, subgraph_labels).mean()
                
                if self.args.pretrain_strategy == 'rm_fp_pred':
                    loss = (sl_loss + md_loss + subgraph_loss)/3
                elif self.args.pretrain_strategy == 'rm_md_pred':
                    loss = (sl_loss + fp_loss + subgraph_loss)/3
                elif self.args.pretrain_strategy == 'rm_both_pred':
                    loss = (sl_loss + subgraph_loss)/2
                elif self.args.pretrain_strategy == 'rm_none_pred':
                    loss = (sl_loss + fp_loss + md_loss + subgraph_loss) / 4
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                self.optimizer.step()
                self.n_updates += 1
                self.training_updates += 1
                self.lr_scheduler.step()
                if self.summary_writer is not None and self.local_rank == 0:
                    loss_mask = self.sl_loss_fn(sl_predictions.detach().cpu()[mask_replace_keep==1],sl_labels.detach().cpu()[mask_replace_keep==1]).mean()
                    loss_replace = self.sl_loss_fn(sl_predictions.detach().cpu()[mask_replace_keep==2],sl_labels.detach().cpu()[mask_replace_keep==2]).mean()
                    loss_keep = self.sl_loss_fn(sl_predictions.detach().cpu()[mask_replace_keep==3],sl_labels.detach().cpu()[mask_replace_keep==3]).mean()
                    preds = np.argmax(sl_predictions.detach().cpu().numpy(),axis=-1)
                    pred_subgraph = np.argmax(subgraph_prediction.detach().cpu().numpy(),axis=-1)
                    labels = sl_labels.detach().cpu().numpy()
                    labels_subgraph = subgraph_labels.detach().cpu().numpy()
                    self.summary_writer.add_scalar('Loss/loss_tot', loss.item(), self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_bert', sl_loss.item(), self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_mask', loss_mask.item(), self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_replace', loss_replace.item(), self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_keep', loss_keep.item(), self.n_updates)
                    
                    if self.args.pretrain_strategy != 'rm_fp_pred':
                        self.summary_writer.add_scalar('Loss/loss_clf', fp_loss.item(), self.n_updates)
                    if self.args.pretrain_strategy != 'rm_md_pred':
                        self.summary_writer.add_scalar('Loss/loss_reg', md_loss.item(), self.n_updates)
                
                    self.summary_writer.add_scalar('Loss/loss_subgraph', subgraph_loss.item(), self.n_updates)
                    self.summary_writer.add_scalar('LR', torch.tensor(self.lr_scheduler.get_lr()[-1]).item(), self.n_updates)
                    
                    self.summary_writer.add_scalar('F1_micro/all', f1_score(preds, labels, average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/all', f1_score(preds, labels, average='macro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_micro/mask', f1_score(preds[mask_replace_keep==1], labels[mask_replace_keep==1], average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/mask', f1_score(preds[mask_replace_keep==1], labels[mask_replace_keep==1], average='macro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_micro/replace', f1_score(preds[mask_replace_keep==2], labels[mask_replace_keep==2], average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/replace', f1_score(preds[mask_replace_keep==2], labels[mask_replace_keep==2], average='macro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_micro/keep', f1_score(preds[mask_replace_keep==3], labels[mask_replace_keep==3], average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/keep', f1_score(preds[mask_replace_keep==3], labels[mask_replace_keep==3], average='macro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_micro/subgraph', f1_score(pred_subgraph, labels_subgraph, average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/subgraph', f1_score(pred_subgraph, labels_subgraph, average='macro'), self.n_updates)
                    self.summary_writer.add_scalar(f'Clf/{self.clf_evaluator.eval_metric}_all', np.mean(self.clf_evaluator.eval(fps, fp_predictions)), self.n_updates)
                if self.n_updates % 1000 == 0:
                    if self.local_rank == 0:
                        print('%d steps finished!' % self.n_updates)               
                if self.training_updates == self.args.n_steps:
                    if self.local_rank == 0:
                        print(f'now the update step is: {self.n_updates}')
                        self.save_model(model)
                    break

            except Exception as e:
                print(e)
            else:
                continue

    def fit(self, model, train_loader, train_episode):
        if self.local_rank == 0:
            if 'chembl' in train_loader.dataset.root_path:
                print('Training on ChEMBL dataset!')
            elif 'pubchem' in train_loader.dataset.root_path:
                print('Training on PubChem dataset!')
            else:
                raise ValueError('Unknown Pretraining dataset!') # type of dataset could be changed here
            
        if train_episode == 1:
            if self.local_rank == 0:
                print(f'training updates:{self.training_updates}, n_updates:{self.n_updates}')
        elif train_episode == 2:
            self.training_updates = 0
            if self.local_rank == 0:
                print(f'training updates:{self.training_updates}, n_updates:{self.n_updates}')
        
        for epoch in range(1, 1001):
            model.train()
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.training_updates >= self.args.n_steps:
                break

    def save_model(self, model):
        torch.save(model.state_dict(), self.args.save_path+f"model_{self.n_updates}.pth")

    