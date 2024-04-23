import os
import wandb
import torch
import torch.nn.functional as F

class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, sl_loss_fn, contrastive_loss_fn,
                 reg_evaluator, clf_evaluator, result_tracker, device, ddp=False, local_rank=1):
        
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.reg_loss_fn = reg_loss_fn
        self.clf_loss_fn = clf_loss_fn
        self.sl_loss_fn = sl_loss_fn
        self.contrastive_loss_fn = contrastive_loss_fn
        self.reg_evaluator = reg_evaluator
        self.clf_evaluator = clf_evaluator
        self.result_tracker = result_tracker
        self.device = device
        self.ddp = ddp
        self.local_rank = local_rank
        self.n_updates = 0      
    
        self.gradient_accumulate_steps = args.gradient_accumulate_steps
        self.training_updates = 0
        self.train_episode = 1 # mark training episode 1 for the first and 2 for the second
    
    def _forward_epoch(self, model, batched_data):
        (smiles, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds) = batched_data
        batched_graph = batched_graph.to(self.device)
        fps = fps.to(self.device)
        mds = mds.to(self.device)
        sl_labels = sl_labels.to(self.device)
        disturbed_fps = disturbed_fps.to(self.device)
        disturbed_mds = disturbed_mds.to(self.device)
        sl_predictions, fp_predictions, md_predictions, z  = model(batched_graph, disturbed_fps, disturbed_mds)
        
        return sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds, z

    def info_nce_loss(self, features):
        T = 0.1
        labels = torch.cat([torch.arange(features.shape[0] // 2) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # broadcast
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device) # positives in logits[:, 0] so the probability should be the max

        logits = logits / T
        return logits, labels
    
    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            
            sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds, z = self._forward_epoch(model, batched_data)
            sl_loss = self.sl_loss_fn(sl_predictions, sl_labels).mean()
            fp_loss = self.clf_loss_fn(fp_predictions, fps).mean()
            md_loss = self.reg_loss_fn(md_predictions, mds).mean()
            
            logits, labels = self.info_nce_loss(z)
            contrastive_loss = self.contrastive_loss_fn(logits, labels).mean()

            loss = (sl_loss + fp_loss + md_loss + contrastive_loss) / 4
            
            if self.gradient_accumulate_steps > 1:
                loss = loss / self.gradient_accumulate_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                self.optimizer.step()
                self.n_updates += 1
                self.training_updates += 1
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            if self.local_rank == 0:
                if self.args.pretrain_strategy == 'rm_fp_pred':
                    wandb.log({'train_loss': loss, 'sl_loss': sl_loss, 'md_loss': md_loss, 'contrastive_loss': contrastive_loss, 'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
                
                elif self.args.pretrain_strategy == 'rm_md_pred':
                    wandb.log({'train_loss': loss, 'sl_loss': sl_loss, 'fp_loss': fp_loss, 'contrastive_loss': contrastive_loss, 'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
                    
                elif self.args.pretrain_strategy == 'rm_both_pred':
                    wandb.log({'train_loss': loss, 'sl_loss': sl_loss, 'contrastive_loss': contrastive_loss, 'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
                    
                elif self.args.pretrain_strategy == 'rm_none_pred':
                    wandb.log({'train_loss': loss, 'sl_loss': sl_loss, 'fp_loss': fp_loss, 'md_loss': md_loss, 'contrastive_loss': contrastive_loss, 'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})
            
            if self.n_updates % 1000 == 0:
                if self.local_rank == 0:
                    print('%d steps finished!' % self.n_updates)               
            if self.training_updates == self.args.n_steps:
                if self.local_rank == 0:
                    print(f'now the update step is: {self.n_updates}')
                    self.save_model(model)
                break

    def fit(self, model, train_loader, train_episode):
        if self.local_rank == 0:
            if 'chembl' in train_loader.dataset.root_path:
                print('Training on ChEMBL dataset!')
            elif 'pubchem' in train_loader.dataset.root_path:
                print('Training on PubChem dataset!')
            elif 'mix' in train_loader.dataset.root_path:
                print('Training on 12M mix dataset!')
            else:
                raise ValueError('Unknown Pretraining dataset!') # type of dataset could be changed here
            
        if train_episode == 1:
            if self.local_rank == 0:
                print(f'training updates:{self.training_updates}, n_updates:{self.n_updates}')
        elif train_episode == 2:
            self.training_updates = 0
            if self.local_rank == 0:
                print(f'training updates:{self.training_updates}, n_updates:{self.n_updates}')
        
        self.optimizer.zero_grad()
        for epoch in range(1, 1001):
            model.train()
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.training_updates >= self.args.n_steps:
                break

    def save_model(self, model):
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        
        if self.args.pretrain2_path == None:
            save_path = os.path.join(self.args.save_path, f"one_stage{self.args.pretrain1_path.replace('/', '-')}_{self.n_updates}_{self.args.pretrain_strategy}.pth")
        else:
            save_path = os.path.join(self.args.save_path, f"two_stage{self.args.pretrain1_path.replace('/', '-')}_{self.n_updates}_{self.args.pretrain_strategy}.pth")
        torch.save(model.state_dict(), save_path)

    