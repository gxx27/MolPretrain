import torch
import numpy as np
from sklearn.metrics import f1_score
from copy import deepcopy

class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, sl_loss_fn, 
                 reg_evaluator, clf_evaluator, result_tracker, summary_writer, device, ddp=False, local_rank=1, 
                 candi_rate=0.15, mask_rate=0.8, replace_rate=0.1, keep_rate=0.1, fp_disturb_rate=0.15, md_disturb_rate=0.15):
        
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
        
        self.candi_rate = candi_rate
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.keep_rate = keep_rate       
        
        self.fp_disturb_rate = fp_disturb_rate
        self.md_disturb_rate = md_disturb_rate
    
    def bert_mask_nodes(self, g):
        n_nodes = g.number_of_nodes()
        all_ids = np.arange(0, n_nodes, 1, dtype=np.int64)
        valid_ids = torch.where(g.ndata['vavn']<=0)[0].numpy()
        valid_labels = g.ndata['label'][valid_ids].numpy()
        probs = np.ones(len(valid_labels))/len(valid_labels)
        unique_labels = np.unique(np.sort(valid_labels))
        for label in unique_labels:
            label_pos = (valid_labels==label)
            probs[label_pos] = probs[label_pos]/np.sum(label_pos)
        probs = probs/np.sum(probs)
        candi_ids = np.random.choice(valid_ids, size=int(len(valid_ids)*self.candi_rate),replace=False, p=probs)

        mask_ids = np.random.choice(candi_ids, size=int(len(candi_ids)*self.mask_rate),replace=False)
        
        candi_ids = np.setdiff1d(candi_ids, mask_ids)
        replace_ids = np.random.choice(candi_ids, size=int(len(candi_ids)*(self.replace_rate/(1-self.keep_rate))),replace=False)
        
        keep_ids = np.setdiff1d(candi_ids, replace_ids)

        g.ndata['mask'] = torch.zeros(n_nodes,dtype=torch.long)
        g.ndata['mask'][mask_ids] = 1
        g.ndata['mask'][replace_ids] = 2
        g.ndata['mask'][keep_ids] = 3
        sl_labels = g.ndata['label'][g.ndata['mask']>=1].clone()

        # Pre-replace
        new_ids = np.random.choice(valid_ids, size=len(replace_ids),replace=True, p=probs)
        replace_labels = g.ndata['label'][replace_ids].numpy()
        new_labels = g.ndata['label'][new_ids].numpy()
        is_equal = (replace_labels == new_labels)
        while(np.sum(is_equal)):
            new_ids[is_equal] = np.random.choice(valid_ids, size=np.sum(is_equal),replace=True, p=probs)
            new_labels = g.ndata['label'][new_ids].numpy()
            is_equal = (replace_labels == new_labels)
        g.ndata['begin_end'][replace_ids] = g.ndata['begin_end'][new_ids].clone()
        g.ndata['edge'][replace_ids] = g.ndata['edge'][new_ids].clone()
        g.ndata['vavn'][replace_ids] = g.ndata['vavn'][new_ids].clone()
        return sl_labels
    
    def disturb_md(self, md):
        md = deepcopy(md)
        b, d = md.shape
        md = md.reshape(-1)
        sampled_ids = np.random.choice(b*d, int(b*d*self.md_disturb_rate), replace=False)
        a = torch.empty(len(sampled_ids)).uniform_(0, 1)
        sampled_md = a
        md[sampled_ids] = sampled_md
        return md.reshape(b,d)

    def _forward_epoch(self, model, batched_data): # 这个函数实现动态掩码机制
        # (smiles, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds) = batched_data
        (smiles, batched_graph, fps, mds) = batched_data # dynamic masking changed here
        sl_labels = self.bert_mask_nodes(batched_graph).to(self.device) # dynamic mask node, fp and md
        disturbed_mds = self.disturb_md(mds).to(self.device)
        batched_graph = batched_graph.to(self.device)
        fps = fps.to(self.device)
        mds = mds.to(self.device)
        disturbed_fps = fps.clone()
        sl_predictions, fp_predictions, md_predictions = model(batched_graph, disturbed_fps, disturbed_mds)
        mask_replace_keep = batched_graph.ndata['mask'][batched_graph.ndata['mask']>=1].cpu().numpy()
        return mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds
    
    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            try:
                self.optimizer.zero_grad()
                mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds = self._forward_epoch(model, batched_data)
                sl_loss = self.sl_loss_fn(sl_predictions, sl_labels).mean()
                # fp_loss = self.clf_loss_fn(fp_predictions, fps).mean()
                md_loss = self.reg_loss_fn(md_predictions, mds).mean()
                loss = (sl_loss + md_loss)/2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                self.optimizer.step()
                self.n_updates += 1
                self.lr_scheduler.step()
                if self.summary_writer is not None:
                    loss_mask = self.sl_loss_fn(sl_predictions.detach().cpu()[mask_replace_keep==1],sl_labels.detach().cpu()[mask_replace_keep==1]).mean()
                    loss_replace = self.sl_loss_fn(sl_predictions.detach().cpu()[mask_replace_keep==2],sl_labels.detach().cpu()[mask_replace_keep==2]).mean()
                    loss_keep = self.sl_loss_fn(sl_predictions.detach().cpu()[mask_replace_keep==3],sl_labels.detach().cpu()[mask_replace_keep==3]).mean()
                    preds = np.argmax(sl_predictions.detach().cpu().numpy(),axis=-1)
                    labels = sl_labels.detach().cpu().numpy()
                    self.summary_writer.add_scalar('Loss/loss_tot', loss.item(), self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_bert', sl_loss.item(), self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_mask', loss_mask.item(), self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_replace', loss_replace.item(), self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_keep', loss_keep.item(), self.n_updates)
                    # self.summary_writer.add_scalar('Loss/loss_clf', fp_loss.item(), self.n_updates)
                    self.summary_writer.add_scalar('Loss/loss_reg', md_loss.item(), self.n_updates)
                    
                    self.summary_writer.add_scalar('F1_micro/all', f1_score(preds, labels, average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/all', f1_score(preds, labels, average='macro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_micro/mask', f1_score(preds[mask_replace_keep==1], labels[mask_replace_keep==1], average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/mask', f1_score(preds[mask_replace_keep==1], labels[mask_replace_keep==1], average='macro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_micro/replace', f1_score(preds[mask_replace_keep==2], labels[mask_replace_keep==2], average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/replace', f1_score(preds[mask_replace_keep==2], labels[mask_replace_keep==2], average='macro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_micro/keep', f1_score(preds[mask_replace_keep==3], labels[mask_replace_keep==3], average='micro'), self.n_updates)
                    self.summary_writer.add_scalar('F1_macro/keep', f1_score(preds[mask_replace_keep==3], labels[mask_replace_keep==3], average='macro'), self.n_updates)
                    self.summary_writer.add_scalar(f'Clf/{self.clf_evaluator.eval_metric}_all', np.mean(self.clf_evaluator.eval(fps, fp_predictions)), self.n_updates)
                if self.n_updates % 1000 == 0:
                    if self.local_rank == 0:
                        print('%d steps finished!' % self.n_updates)               
                if self.n_updates % self.args.n_steps == 0:
                    if self.local_rank == 0:
                        print(f'now the update step is: {self.n_updates}')
                        self.save_model(model)
                    break

            except Exception as e:
                print(e)
            else:
                continue

    def fit(self, model, train_loader):
        if self.local_rank == 0:
            if 'chembl' in train_loader.dataset.root_path:
                print('Training on ChEMBL dataset!')
            elif 'pubchem' in train_loader.dataset.root_path:
                print('Training on PubChem dataset!')
            else:
                raise ValueError('Unknown Pretraining dataset!') # type of dataset could be changed here
        
        for epoch in range(1, 1001):
            model.train()
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.n_updates % self.args.n_steps == 0:
                break

    def save_model(self, model):
        torch.save(model.state_dict(), self.args.save_path+f"model_{self.n_updates}.pth")

    