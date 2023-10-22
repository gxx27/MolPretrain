import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from dataset_test import MolTestDatasetWrapper
from gtn_finetune import GTN
import argparse
import matplotlib.pyplot as plt
# import wandb

import warnings

warnings.filterwarnings("ignore")


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class Model(nn.Module):   
    def __init__(self, GTN, task, node_dim, finetune=False):
        super(Model, self).__init__()
        self.gnn = GTN
        self.task = task
        self.node_dim = node_dim
        self.finetune = finetune

        # if self.finetune:
        #     for _, parms in gnn.named_parameters():
        #         parms.requires_grad = False

        if self.task == 'classification':
            self.pred_head = nn.Sequential(
                    nn.Linear(self.node_dim, self.node_dim//2), 
                    nn.Softplus(),
                    nn.Linear(self.node_dim//2, 2),
                )

        elif task == 'regression':
            self.pred_head = nn.Sequential(
                nn.Linear(self.node_dim, self.node_dim//2), 
                nn.Softplus(),
                nn.Linear(self.node_dim//2, 1)
            )
    
    def forward(self, data):
        gnn_output = self.gnn(data)
        return self.pred_head(gnn_output)

def eval(model, dataloader, loss_func, task, normalizer=None):
    predictions = []
    labels = []
    with torch.no_grad():
        model.eval()
        eval_loss = 0.0
        num_data = 0

        for bn, data in enumerate(dataloader):
            data = data.to(device)
            pred = model(data)

            if task == 'classification':
                loss = loss_func(pred, data['atom'].y.flatten())
            elif task == 'regression':
                if normalizer:
                    loss = loss_func(pred, normalizer.norm(data['atom'].y))
                else:
                    loss = loss_func(pred, data['atom'].y)

            eval_loss += loss.item() * data['atom'].y.size(0)
            num_data += data['atom'].y.size(0)

            if normalizer:
                pred = normalizer.denorm(pred)

            if task == 'classification':
                pred = F.softmax(pred, dim=-1)
            predictions.extend(pred.cpu().detach().numpy())
            labels.extend(data['atom'].y.cpu().flatten().numpy())

        eval_loss /= num_data

    if task == 'regression':
        predictions = np.array(predictions)
        labels = np.array(labels)
        if task_name in ['qm7', 'qm8', 'qm9']:
            mae = mean_absolute_error(labels, predictions)
            result = mae

        else:
            rmse = mean_squared_error(labels, predictions, squared=False)
            result = rmse            

    elif task == 'classification':
        predictions = np.array(predictions)
        labels = np.array(labels)
        roc_auc = roc_auc_score(labels, predictions[:,1])
        result = roc_auc

    model.train()
    return eval_loss, result

def result_tracker(task, result, best_result):
    if task == 'regression':
        if result < best_result:
            return True
    
    elif task == 'classification':
        if result > best_result:
            return True

def draw(loss, result, task_name, target, data):
    plt.figure()

    if data == 'train':
        plt.plot(range(len(loss)), loss, label='Training Loss')
        xlabel = 'Epochs'
        title = f'Training Loss on task {task_name}'
        save_path = f'{task_name}_train_loss_{target}.png'

    elif data == 'valid':
        plt.plot(range(len(loss)), loss, label='Validation Loss')
        xlabel = 'Epochs'
        title = f'Validation Loss on task {task_name}'
        save_path = f'{task_name}_val_loss_{target}.png'

    elif data == 'test':
        plt.plot(range(len(loss)), loss, label='Test Loss')
        xlabel = 'Epochs'
        title = f'Test Loss on task {task_name}'
        save_path = f'{task_name}_test_loss_{target}.png'

    else:
        raise TypeError("Invalid data type")

    plt.xlabel(xlabel)
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)

    if task == 'regression':
        if task_name in ['qm7', 'qm8', 'qm9']:
            label = 'MAE'
            save_path = f'{task_name}_{data}_mae_{target}.png'
        else:
            label = 'RMSE'
            save_path = f'{task_name}_{data}_rmse_{target}.png'

    elif task == 'classification':
        label = 'ROC_AUC'
        save_path = f'{task_name}_{data}_roc_auc_{target}.png'

    plt.figure()
    plt.plot(range(len(result)), result, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(label)
    plt.title(f'{label} on target {target}')
    plt.legend()
    plt.savefig(save_path)

def process_dataset(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip().split(',')
        filtered_line = [item.strip() for item in first_line if item.strip() != 'smiles']
        result = [f'{item}' for item in filtered_line]
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Training Epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='number of workers')
    parser.add_argument('--valid', type=float, default=0.05,
                        help='data size for validation')
    parser.add_argument('--path', type=str, default='/data1/gx/datasets/',
                        help='dataset path')
    parser.add_argument('--dataset', type=str, default='bbbp',
                        help='dataset name')
    parser.add_argument('--split', type=str, default='scaffold',
                        help='the way to split dataset')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='hidden dimensions')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--init_lr', type=float, default=3e-5,
                        help='learning rate')
    parser.add_argument('--base_lr', type=float, default=1e-5,
                        help='learning rate')    
    parser.add_argument('--weight_decay', type=str, default='1e-6',
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GT/FastGT layers')
    parser.add_argument("--channel_agg", type=str, default='concat')
    parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")

    args = parser.parse_args()
    print(args)

    epochs = args.epoch
    batch_size = args.batch
    num_workers = args.num_workers
    valid_size = args.valid
    dataset = args.dataset
    dataset_path = args.path
    dataset_split = args.split
    node_dim = args.node_dim
    emb_dim = args.emb_dim
    num_channels = args.num_channels
    init_lr = args.init_lr
    base_lr = args.base_lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # datasets = ['BBBP', 'Tox21', 'ClinTox', 'HIV', 'BACE', 'SIDER', 'MUV', 'FreeSolv', 'ESOL', 'Lipo', 'qm7', 'qm8', 'qm9']

    # wandb.login()
    # wandb.init()

    if dataset == 'bbbp':
        task = 'classification'
        task_name = 'bbbp'
        path = dataset_path + 'bbbp/bbbp.csv'
        target_list = process_dataset(path)

    elif dataset == 'tox21':
        task = 'classification'
        task_name = 'tox21'
        path = dataset_path + 'tox21/tox21.csv'
        target_list = process_dataset(path)

    elif dataset == 'clintox':
        task = 'classification'
        task_name = 'clintox'
        path = dataset_path + 'clintox/clintox.csv'
        target_list = process_dataset(path)

    elif dataset == 'hiv':
        task = 'classification'
        task_name = 'hiv'
        path = dataset_path + 'hiv/hiv.csv'
        target_list = process_dataset(path)

    elif dataset == 'bace':
        task = 'classification'
        task_name = 'bace'
        path = dataset_path + 'bace/bace.csv'
        target_list = process_dataset(path)

    elif dataset == 'sider':
        task = 'classification'
        task_name = 'sider'
        path = dataset_path + 'sider/sider.csv'
        # target_list = process_dataset(path)
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif dataset == 'muv':
        task = 'classification'
        task_name = 'muv'
        path = dataset_path + 'muv/muv.csv'
        target_list = process_dataset(path)

    elif dataset == 'freesolv':
        task = 'regression'
        task_name = 'freesolv'
        path = dataset_path + 'freesolv/freesolv.csv'
        target_list = process_dataset(path)
    
    elif dataset == 'esol':
        task = 'regression'
        task_name = 'esol'
        path = dataset_path + 'esol/esol.csv'
        target_list = process_dataset(path)

    elif dataset == 'estrogen':
        task = 'regression'
        task_name = 'estrogen'
        path = dataset_path + 'estrogen/estrogen.csv'
        target_list = process_dataset(path)

    elif dataset == 'metstab':
        task = 'regression'
        task_name = 'metstab'
        path = dataset_path + 'metstab/metstab.csv'
        target_list = process_dataset(path)

    elif dataset == 'lipo':
        task = 'regression'
        task_name = 'lipo'
        path = dataset_path + 'lipo/lipo.csv'
        target_list = process_dataset(path)
    
    elif dataset == 'qm7':
        task = 'regression'
        task_name = 'qm7'
        path = dataset_path + 'qm7/qm7.csv'
        target_list = process_dataset(path)

    elif dataset == 'qm8':
        task = 'regression'
        task_name = 'qm8'
        path = dataset_path + 'qm8/qm8.csv'
        target_list = process_dataset(path)
    
    elif dataset == 'qm9':
        task = 'regression'
        task_name = 'qm9'
        path = dataset_path + 'qm9/qm9.csv'
        target_list = process_dataset(path)

    else:
        raise ValueError('Undefined downstream task!')

    if task == 'classification':
        loss_func = nn.CrossEntropyLoss()
    elif task == 'regression':
        if task_name in ['qm7', 'qm8', 'qm9']:
            loss_func = nn.L1Loss()
        else:
            loss_func = nn.MSELoss()

    gnn = GTN(
        task = task,
        num_channels = num_channels,
        w_in = emb_dim,
        w_out = node_dim,
        num_layers = num_layers,
        emb_dim = emb_dim,
        args = args
    )

    gnn.load_state_dict(torch.load('model.pth'))

    model = Model(gnn, task, node_dim, finetune=True)

    model.to(device)

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    # wandb.watch(model)
    for target in target_list:
        dataset = MolTestDatasetWrapper(batch_size=batch_size, num_workers=4, valid_size=0.1, test_size=0.1, data_path=path, target=target,task=task, splitting=dataset_split)
        train_loader, valid_loader, test_loader = dataset.get_data_loaders()

        print(f'Working on dataset {task_name} with target {target}')

        if task_name in ['qm7', 'qm9']:
            labels = []
            for _ , d in enumerate(train_loader):
                labels.append(d['atom'].y)
            labels = torch.cat(labels)
            normalizer = Normalizer(labels)
            print(normalizer.mean, normalizer.std, labels.shape)
        
        else:
            normalizer = None

        train_loss = []
        train_auc = []
        train_mae = []
        train_rmse = []

        val_loss = []
        val_auc = []
        val_mae = []
        val_rmse = []

        t_loss = []
        t_auc = []
        t_mae = []
        t_rmse = []

        if task == 'classification':
            best_result = 0.0
        elif task == 'regression':
            best_result = 999.0

        # training
        for epoch in range(epochs):
            total_loss = 0
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(device)

                pred = model(data)

                if task == 'classification':
                    loss = loss_func(pred, data['atom'].y.flatten())
                elif task == 'regression':
                    if normalizer:
                        loss = loss_func(pred, normalizer.norm(data['atom'].y))
                    else:
                        loss = loss_func(pred, data['atom'].y)

                loss.backward()
                optimizer.step()
                # wandb.log({'batch_loss':loss.item()})
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            train_loss.append(avg_train_loss)

            _, train_result = eval(model, train_loader, loss_func, task, normalizer)
            if task == 'regression':
                if task_name in ['qm7', 'qm8', 'qm9']:
                    train_mae.append(train_result)
                else:
                    train_rmse.append(train_result)
            
            elif task == 'classification':
                train_auc.append(train_result)
            # wandb.log({'metric':avg_train_loss,'lr':optimizer.param_groups[0]['lr']})

            # validation & test
            valid_loss, valid_result = eval(model, valid_loader, loss_func, task, normalizer)
            if task == 'regression':
                if task_name in ['qm7', 'qm8', 'qm9']:
                    print('Validation loss:', valid_loss, 'MAE:', valid_result)
                    val_loss.append(valid_loss)
                    val_mae.append(valid_result)
                else:
                    print('Validation loss:', valid_loss, 'RMSE:', valid_result)
                    val_loss.append(valid_loss)
                    val_rmse.append(valid_result) 
            
            elif task == 'classification':
                print('Validation loss:', valid_loss, 'ROC AUC:', valid_result)
                val_loss.append(valid_loss)
                val_auc.append(valid_result)

            # test
            test_loss, test_result = eval(model, test_loader, loss_func, task, normalizer)
            if task == 'regression':
                if task_name in ['qm7', 'qm8', 'qm9']:
                    print('Test loss:', test_loss, 'MAE:', test_result)
                    t_loss.append(test_loss)
                    t_mae.append(test_result)

                else:
                    print('Test loss:', test_loss, 'RMSE:', test_result)
                    t_loss.append(test_loss)
                    t_rmse.append(test_result)

            
            elif task == 'classification':
                print('Test loss:', test_loss, 'ROC AUC:', test_result)
                t_loss.append(test_loss)
                t_auc.append(test_result)

            # track
            if result_tracker(task, valid_result, best_result): # early stopping
                best_result = valid_result
                best_train_result = train_result
                best_valid_result = valid_result
                best_test_result = test_result
                best_epoch = epoch
            
            if epoch - best_epoch >= 20:
                print('Train: %.2f, Valid: %.2f, Test: %.2f' % (best_train_result, best_valid_result, best_test_result))
                break

        if task == 'regression':
            if task_name in ['qm7', 'qm8', 'qm9']:
                draw(train_loss, train_mae, task_name, target, data='train')
                draw(val_loss, val_mae, task_name, target, data='valid')
                draw(t_loss, t_mae, task_name, target, data='test')
            else:
                draw(train_loss, train_rmse, task_name, target, data='train')
                draw(val_loss, val_rmse, task_name, target, data='valid')
                draw(t_loss, t_rmse, task_name, target, data='test')
        
        elif task == 'classification':
            draw(train_loss, train_auc, task_name, target, data='train')
            draw(val_loss, val_auc, task_name, target, data='valid')
            draw(t_loss, t_auc, task_name, target, data='test')
        else:
            raise TypeError
        
        result = {'task_name': [], 'metric': []}
        result['task_name'].append((task_name, target))
        if task == 'regression':
            if task_name in ['qm7', 'qm8', 'qm9']:
                result['metric'].append(('MAE', best_test_result))
            else:
                result['metric'].append(('RMSE', best_test_result))
        elif task == 'classification':
            result['metric'].append(('ROC_AUC', best_test_result))

        df = pd.DataFrame(result)

        csv_file = f'{task_name}_test_results.csv' # results saving path
        df.to_csv(csv_file, mode='a+', index=False)
