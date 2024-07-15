import sys
sys.path.append('..')

import numpy as np
import os
import random
import argparse
import wandb

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data.distributed import DistributedSampler

from src.utils import set_random_seed
from src.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES
from src.data.pretrain_dataset import MoleculeDataset
from src.data.collator import Collator_pretrain
from src.model.light import LiGhTPredictor as LiGhT
from src.trainer.scheduler import PolynomialDecayLR
from src.trainer.pretrain_trainer import Trainer
from src.trainer.evaluator import Evaluator
from src.trainer.result_tracker import Result_Tracker
from src.model_config import config_dict
import warnings
warnings.filterwarnings("ignore")
local_rank = int(os.environ['LOCAL_RANK'])


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training LiGhT")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--pretrain1_path", type=str, default=None)
    parser.add_argument("--pretrain2_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default='../models')
    parser.add_argument("--n_steps", type=int, default=100000)
    parser.add_argument("--total_steps", type=int, default=300000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--gradient_accumulate_steps", type=int, default=2)
    parser.add_argument("--config", type=str, default="KG-GCPT")
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--n_devices", type=int, default=1)
    parser.add_argument("--data_aug1", type=str, default=None, help='choose from drop_nodes, permute_edges, mask_nodes, subgraph')
    parser.add_argument("--data_aug1_rate", type=float, default=0.2)
    parser.add_argument("--data_aug2", type=str, default=None, help='choose from drop_nodes, permute_edges, mask_nodes, subgraph')
    parser.add_argument("--data_aug2_rate", type=float, default=0.2)
    parser.add_argument("--save_name", type=str, default=None, help='name of saved model')
    parser.add_argument("--wandb_key", type=str, default=None)
    args = parser.parse_args()
    return args

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    args = parse_args()
    config = config_dict[args.config]
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device('cuda', local_rank)
    set_random_seed(args.seed)
    print(local_rank)
    val_results, test_results, train_results = [], [], []
    
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)        
    collator = Collator_pretrain(
        vocab, max_length=config['path_length'], n_virtual_nodes=2, 
        candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'], 
        data_aug1=args.data_aug1, data_aug1_rate=args.data_aug1_rate, data_aug2=args.data_aug2, data_aug2_rate=args.data_aug2_rate
    )
    train_dataset = MoleculeDataset(root_path=args.pretrain1_path)
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset), 
                              batch_size=args.batch_size// args.n_devices, num_workers=args.n_threads, 
                              worker_init_fn=seed_worker, drop_last=True, collate_fn=collator, pin_memory=True
    )

    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=train_dataset.d_fps,
        d_md_feats=train_dataset.d_mds,
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=config['input_drop'],
        attn_drop=config['attn_drop'],
        feat_drop=config['feat_drop'],
        n_node_types=vocab.vocab_size
    ).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=20000, tot_updates=args.total_steps,lr=config['lr'], end_lr=1e-9,power=1)
    reg_loss_fn = MSELoss(reduction='none')
    clf_loss_fn = BCEWithLogitsLoss(weight=train_dataset._task_pos_weights.to(device),reduction='none')
    sl_loss_fn = CrossEntropyLoss(reduction='none')
    contrastive_loss_fn = CrossEntropyLoss().to(device)
    reg_metric, clf_metric = "r2", "rocauc_resp"
    reg_evaluator = Evaluator("mix", reg_metric, train_dataset.d_mds)
    clf_evaluator = Evaluator("mix", clf_metric, train_dataset.d_fps)
    result_tracker = Result_Tracker(reg_metric)
    
    if local_rank == 0:
        os.environ["WANDB_PROJECT"] = "KPGT"
        os.environ["WANDB_SILENT"] = "true"
        wandb.login(key=args.wandb_key)
        wandb.init()
        wandb.watch(model)

    trainer = Trainer(args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, sl_loss_fn, contrastive_loss_fn,
                      reg_evaluator, clf_evaluator, result_tracker, device=device,local_rank=local_rank)
    trainer.fit(model, train_loader, train_episode=1)

    del train_dataset # reduce memory cost
    del train_loader
    
    if 'mix' not in args.pretrain1_path and not args.pretrain2_path == None:
        train_dataset = MoleculeDataset(root_path=args.pretrain2_path)
        train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset), batch_size=args.batch_size// args.n_devices, 
                                  num_workers=args.n_threads, worker_init_fn=seed_worker, drop_last=True, collate_fn=collator, pin_memory=True
        )

        clf_loss_fn = BCEWithLogitsLoss(weight=train_dataset._task_pos_weights.to(device),reduction='none')
            
        trainer.fit(model, train_loader, train_episode=2)