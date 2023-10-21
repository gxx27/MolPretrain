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

    # elif dataset == 'toxcast':
    #     task = 'regression'
    #     task_name = 'toxcast'
    #     path = dataset_path + 'toxcast/toxcast.csv'
    #     target_list = [
    #         'ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive', 'APR_HepG2_CellCycleArrest_24h_dn',
    #         'APR_HepG2_CellCycleArrest_24h_up', 'APR_HepG2_CellCycleArrest_72h_dn', 'APR_HepG2_CellLoss_24h_dn', 
    #         'APR_HepG2_CellLoss_72h_dn', 'APR_HepG2_MicrotubuleCSK_24h_dn', 'APR_HepG2_MicrotubuleCSK_24h_up',
    #         'APR_HepG2_MicrotubuleCSK_72h_dn', 'APR_HepG2_MicrotubuleCSK_72h_up', 'APR_HepG2_MitoMass_24h_dn',
    #         'APR_HepG2_MitoMass_24h_up', 'APR_HepG2_MitoMass_72h_dn', 'APR_HepG2_MitoMass_72h_up',
    #         'APR_HepG2_MitoMembPot_1h_dn', 'APR_HepG2_MitoMembPot_24h_dn', 'APR_HepG2_MitoMembPot_72h_dn',
    #         'APR_HepG2_MitoticArrest_24h_up', 'APR_HepG2_MitoticArrest_72h_up', 'APR_HepG2_NuclearSize_24h_dn',
    #         'APR_HepG2_NuclearSize_72h_dn', 'APR_HepG2_NuclearSize_72h_up', 'APR_HepG2_OxidativeStress_24h_up',
    #         'APR_HepG2_OxidativeStress_72h_up', 'APR_HepG2_StressKinase_1h_up', 'APR_HepG2_StressKinase_24h_up',
    #         'APR_HepG2_StressKinase_72h_up', 'APR_HepG2_p53Act_24h_up', 'APR_HepG2_p53Act_72h_up',
    #         'APR_Hepat_Apoptosis_24hr_up', 'APR_Hepat_Apoptosis_48hr_up', 'APR_Hepat_CellLoss_24hr_dn',
    #         'APR_Hepat_CellLoss_48hr_dn', 'APR_Hepat_DNADamage_24hr_up', 'APR_Hepat_DNADamage_48hr_up', 
    #         'APR_Hepat_DNATexture_24hr_up', 'APR_Hepat_DNATexture_48hr_up', 'APR_Hepat_MitoFxnI_1hr_dn', 
    #         'APR_Hepat_MitoFxnI_24hr_dn', 'APR_Hepat_MitoFxnI_48hr_dn', 'APR_Hepat_NuclearSize_24hr_dn',
    #         'APR_Hepat_NuclearSize_48hr_dn', 'APR_Hepat_Steatosis_24hr_up', 'APR_Hepat_Steatosis_48hr_up',
    #         'ATG_AP_1_CIS_dn', 'ATG_AP_1_CIS_up', 'ATG_AP_2_CIS_dn',
    #         'ATG_AP_2_CIS_up', 'ATG_AR_TRANS_dn', 'ATG_AR_TRANS_up',
    #         'ATG_Ahr_CIS_dn', 'ATG_Ahr_CIS_up', 'ATG_BRE_CIS_dn',
    #         'ATG_BRE_CIS_up', 'ATG_CAR_TRANS_dn', 'ATG_CAR_TRANS_up',
    #         'ATG_CMV_CIS_dn', 'ATG_CMV_CIS_up', 'ATG_CRE_CIS_dn',
    #         'ATG_CRE_CIS_up', 'ATG_C_EBP_CIS_dn', 'ATG_C_EBP_CIS_up',
    #         'ATG_DR4_LXR_CIS_dn', 'ATG_DR4_LXR_CIS_up', 'ATG_DR5_CIS_dn',
    #         'ATG_DR5_CIS_up', 'ATG_E2F_CIS_dn', 'ATG_E2F_CIS_up',
    #         'ATG_EGR_CIS_up', 'ATG_ERE_CIS_dn', 'ATG_ERE_CIS_up',
    #         'ATG_ERRa_TRANS_dn', 'ATG_ERRg_TRANS_dn', 'ATG_ERRg_TRANS_up',
    #         'ATG_ERa_TRANS_up', 'ATG_E_Box_CIS_dn', 'ATG_E_Box_CIS_up',
    #         'ATG_Ets_CIS_dn', 'ATG_Ets_CIS_up', 'ATG_FXR_TRANS_up',
    #         'ATG_FoxA2_CIS_dn', 'ATG_FoxA2_CIS_up', 'ATG_FoxO_CIS_dn',
    #         'ATG_FoxO_CIS_up', 'ATG_GAL4_TRANS_dn', 'ATG_GATA_CIS_dn',
    #         'ATG_GATA_CIS_up', 'ATG_GLI_CIS_dn', 'ATG_GLI_CIS_up',
    #         'ATG_GRE_CIS_dn', 'ATG_GRE_CIS_up', 'ATG_GR_TRANS_dn',
    #         'ATG_GR_TRANS_up', 'ATG_HIF1a_CIS_dn', 'ATG_HIF1a_CIS_up',
    #         'ATG_HNF4a_TRANS_dn', 'ATG_HNF4a_TRANS_up', 'ATG_HNF6_CIS_dn',
    #         'ATG_HNF6_CIS_up', 'ATG_HSE_CIS_dn', 'ATG_HSE_CIS_up',
    #         'ATG_IR1_CIS_dn', 'ATG_IR1_CIS_up', 'ATG_ISRE_CIS_dn', ATG_ISRE_CIS_up,ATG_LXRa_TRANS_dn,ATG_LXRa_TRANS_up,ATG_LXRb_TRANS_dn,ATG_LXRb_TRANS_up,ATG_MRE_CIS_up,ATG_M_06_TRANS_up,ATG_M_19_CIS_dn,ATG_M_19_TRANS_dn,ATG_M_19_TRANS_up,ATG_M_32_CIS_dn,ATG_M_32_CIS_up,ATG_M_32_TRANS_dn,ATG_M_32_TRANS_up,ATG_M_61_TRANS_up,ATG_Myb_CIS_dn,ATG_Myb_CIS_up,ATG_Myc_CIS_dn,ATG_Myc_CIS_up,ATG_NFI_CIS_dn,ATG_NFI_CIS_up,ATG_NF_kB_CIS_dn,ATG_NF_kB_CIS_up,ATG_NRF1_CIS_dn,ATG_NRF1_CIS_up,ATG_NRF2_ARE_CIS_dn,ATG_NRF2_ARE_CIS_up,ATG_NURR1_TRANS_dn,ATG_NURR1_TRANS_up,ATG_Oct_MLP_CIS_dn,ATG_Oct_MLP_CIS_up,ATG_PBREM_CIS_dn,ATG_PBREM_CIS_up,ATG_PPARa_TRANS_dn,ATG_PPARa_TRANS_up,ATG_PPARd_TRANS_up,ATG_PPARg_TRANS_up,ATG_PPRE_CIS_dn,ATG_PPRE_CIS_up,ATG_PXRE_CIS_dn,ATG_PXRE_CIS_up,ATG_PXR_TRANS_dn,ATG_PXR_TRANS_up,ATG_Pax6_CIS_up,ATG_RARa_TRANS_dn,ATG_RARa_TRANS_up,ATG_RARb_TRANS_dn,ATG_RARb_TRANS_up,ATG_RARg_TRANS_dn,ATG_RARg_TRANS_up,ATG_RORE_CIS_dn,ATG_RORE_CIS_up,ATG_RORb_TRANS_dn,ATG_RORg_TRANS_dn,ATG_RORg_TRANS_up,ATG_RXRa_TRANS_dn,ATG_RXRa_TRANS_up,ATG_RXRb_TRANS_dn,ATG_RXRb_TRANS_up,ATG_SREBP_CIS_dn,ATG_SREBP_CIS_up,ATG_STAT3_CIS_dn,ATG_STAT3_CIS_up,ATG_Sox_CIS_dn,ATG_Sox_CIS_up,ATG_Sp1_CIS_dn,ATG_Sp1_CIS_up,ATG_TAL_CIS_dn,ATG_TAL_CIS_up,ATG_TA_CIS_dn,ATG_TA_CIS_up,ATG_TCF_b_cat_CIS_dn,ATG_TCF_b_cat_CIS_up,ATG_TGFb_CIS_dn,ATG_TGFb_CIS_up,ATG_THRa1_TRANS_dn,ATG_THRa1_TRANS_up,ATG_VDRE_CIS_dn,ATG_VDRE_CIS_up,ATG_VDR_TRANS_dn,ATG_VDR_TRANS_up,ATG_XTT_Cytotoxicity_up,ATG_Xbp1_CIS_dn,ATG_Xbp1_CIS_up,ATG_p53_CIS_dn,ATG_p53_CIS_up,BSK_3C_Eselectin_down,BSK_3C_HLADR_down,BSK_3C_ICAM1_down,BSK_3C_IL8_down,BSK_3C_MCP1_down,BSK_3C_MIG_down,BSK_3C_Proliferation_down,BSK_3C_SRB_down,BSK_3C_Thrombomodulin_down,BSK_3C_Thrombomodulin_up,BSK_3C_TissueFactor_down,BSK_3C_TissueFactor_up,BSK_3C_VCAM1_down,BSK_3C_Vis_down,BSK_3C_uPAR_down,BSK_4H_Eotaxin3_down,BSK_4H_MCP1_down,BSK_4H_Pselectin_down,BSK_4H_Pselectin_up,BSK_4H_SRB_down,BSK_4H_VCAM1_down,BSK_4H_VEGFRII_down,BSK_4H_uPAR_down,BSK_4H_uPAR_up,BSK_BE3C_HLADR_down,BSK_BE3C_IL1a_down,BSK_BE3C_IP10_down,BSK_BE3C_MIG_down,BSK_BE3C_MMP1_down,BSK_BE3C_MMP1_up,BSK_BE3C_PAI1_down,BSK_BE3C_SRB_down,BSK_BE3C_TGFb1_down,BSK_BE3C_tPA_down,BSK_BE3C_uPAR_down,BSK_BE3C_uPAR_up,BSK_BE3C_uPA_down,BSK_CASM3C_HLADR_down,BSK_CASM3C_IL6_down,BSK_CASM3C_IL6_up,BSK_CASM3C_IL8_down,BSK_CASM3C_LDLR_down,BSK_CASM3C_LDLR_up,BSK_CASM3C_MCP1_down,BSK_CASM3C_MCP1_up,BSK_CASM3C_MCSF_down,BSK_CASM3C_MCSF_up,BSK_CASM3C_MIG_down,BSK_CASM3C_Proliferation_down,BSK_CASM3C_Proliferation_up,BSK_CASM3C_SAA_down,BSK_CASM3C_SAA_up,BSK_CASM3C_SRB_down,BSK_CASM3C_Thrombomodulin_down,BSK_CASM3C_Thrombomodulin_up,BSK_CASM3C_TissueFactor_down,BSK_CASM3C_VCAM1_down,BSK_CASM3C_VCAM1_up,BSK_CASM3C_uPAR_down,BSK_CASM3C_uPAR_up,BSK_KF3CT_ICAM1_down,BSK_KF3CT_IL1a_down,BSK_KF3CT_IP10_down,BSK_KF3CT_IP10_up,BSK_KF3CT_MCP1_down,BSK_KF3CT_MCP1_up,BSK_KF3CT_MMP9_down,BSK_KF3CT_SRB_down,BSK_KF3CT_TGFb1_down,BSK_KF3CT_TIMP2_down,BSK_KF3CT_uPA_down,BSK_LPS_CD40_down,BSK_LPS_Eselectin_down,BSK_LPS_Eselectin_up,BSK_LPS_IL1a_down,BSK_LPS_IL1a_up,BSK_LPS_IL8_down,BSK_LPS_IL8_up,BSK_LPS_MCP1_down,BSK_LPS_MCSF_down,BSK_LPS_PGE2_down,BSK_LPS_PGE2_up,BSK_LPS_SRB_down,BSK_LPS_TNFa_down,BSK_LPS_TNFa_up,BSK_LPS_TissueFactor_down,BSK_LPS_TissueFactor_up,BSK_LPS_VCAM1_down,BSK_SAg_CD38_down,BSK_SAg_CD40_down,BSK_SAg_CD69_down,BSK_SAg_Eselectin_down,BSK_SAg_Eselectin_up,BSK_SAg_IL8_down,BSK_SAg_IL8_up,BSK_SAg_MCP1_down,BSK_SAg_MIG_down,BSK_SAg_PBMCCytotoxicity_down,BSK_SAg_PBMCCytotoxicity_up,BSK_SAg_Proliferation_down,BSK_SAg_SRB_down,BSK_hDFCGF_CollagenIII_down,BSK_hDFCGF_EGFR_down,BSK_hDFCGF_EGFR_up,BSK_hDFCGF_IL8_down,BSK_hDFCGF_IP10_down,BSK_hDFCGF_MCSF_down,BSK_hDFCGF_MIG_down,BSK_hDFCGF_MMP1_down,BSK_hDFCGF_MMP1_up,BSK_hDFCGF_PAI1_down,BSK_hDFCGF_Proliferation_down,BSK_hDFCGF_SRB_down,BSK_hDFCGF_TIMP1_down,BSK_hDFCGF_VCAM1_down,CEETOX_H295R_11DCORT_dn,CEETOX_H295R_ANDR_dn,CEETOX_H295R_CORTISOL_dn,CEETOX_H295R_DOC_dn,CEETOX_H295R_DOC_up,CEETOX_H295R_ESTRADIOL_dn,CEETOX_H295R_ESTRADIOL_up,CEETOX_H295R_ESTRONE_dn,CEETOX_H295R_ESTRONE_up,CEETOX_H295R_OHPREG_up,CEETOX_H295R_OHPROG_dn,CEETOX_H295R_OHPROG_up,CEETOX_H295R_PROG_up,CEETOX_H295R_TESTO_dn,CLD_ABCB1_48hr,CLD_ABCG2_48hr,CLD_CYP1A1_24hr,CLD_CYP1A1_48hr,CLD_CYP1A1_6hr,CLD_CYP1A2_24hr,CLD_CYP1A2_48hr,CLD_CYP1A2_6hr,CLD_CYP2B6_24hr,CLD_CYP2B6_48hr,CLD_CYP2B6_6hr,CLD_CYP3A4_24hr,CLD_CYP3A4_48hr,CLD_CYP3A4_6hr,CLD_GSTA2_48hr,CLD_SULT2A_24hr,CLD_SULT2A_48hr,CLD_UGT1A1_24hr,CLD_UGT1A1_48hr,NCCT_HEK293T_CellTiterGLO,NCCT_QuantiLum_inhib_2_dn,NCCT_QuantiLum_inhib_dn,NCCT_TPO_AUR_dn,NCCT_TPO_GUA_dn,NHEERL_ZF_144hpf_TERATOSCORE_up,NVS_ADME_hCYP19A1,NVS_ADME_hCYP1A1,NVS_ADME_hCYP1A2,NVS_ADME_hCYP2A6,NVS_ADME_hCYP2B6,NVS_ADME_hCYP2C19,NVS_ADME_hCYP2C9,NVS_ADME_hCYP2D6,NVS_ADME_hCYP3A4,NVS_ADME_hCYP4F12,NVS_ADME_rCYP2C12,NVS_ENZ_hAChE,NVS_ENZ_hAMPKa1,NVS_ENZ_hAurA,NVS_ENZ_hBACE,NVS_ENZ_hCASP5,NVS_ENZ_hCK1D,NVS_ENZ_hDUSP3,NVS_ENZ_hES,NVS_ENZ_hElastase,NVS_ENZ_hFGFR1,NVS_ENZ_hGSK3b,NVS_ENZ_hMMP1,NVS_ENZ_hMMP13,NVS_ENZ_hMMP2,NVS_ENZ_hMMP3,NVS_ENZ_hMMP7,NVS_ENZ_hMMP9,NVS_ENZ_hPDE10,NVS_ENZ_hPDE4A1,NVS_ENZ_hPDE5,NVS_ENZ_hPI3Ka,NVS_ENZ_hPTEN,NVS_ENZ_hPTPN11,NVS_ENZ_hPTPN12,NVS_ENZ_hPTPN13,NVS_ENZ_hPTPN9,NVS_ENZ_hPTPRC,NVS_ENZ_hSIRT1,NVS_ENZ_hSIRT2,NVS_ENZ_hTrkA,NVS_ENZ_hVEGFR2,NVS_ENZ_oCOX1,NVS_ENZ_oCOX2,NVS_ENZ_rAChE,NVS_ENZ_rCNOS,NVS_ENZ_rMAOAC,NVS_ENZ_rMAOAP,NVS_ENZ_rMAOBC,NVS_ENZ_rMAOBP,NVS_ENZ_rabI2C,NVS_GPCR_bAdoR_NonSelective,NVS_GPCR_bDR_NonSelective,NVS_GPCR_g5HT4,NVS_GPCR_gH2,NVS_GPCR_gLTB4,NVS_GPCR_gLTD4,NVS_GPCR_gMPeripheral_NonSelective,NVS_GPCR_gOpiateK,NVS_GPCR_h5HT2A,NVS_GPCR_h5HT5A,NVS_GPCR_h5HT6,NVS_GPCR_h5HT7,NVS_GPCR_hAT1,NVS_GPCR_hAdoRA1,NVS_GPCR_hAdoRA2a,NVS_GPCR_hAdra2A,NVS_GPCR_hAdra2C,NVS_GPCR_hAdrb1,NVS_GPCR_hAdrb2,NVS_GPCR_hAdrb3,NVS_GPCR_hDRD1,NVS_GPCR_hDRD2s,NVS_GPCR_hDRD4.4,NVS_GPCR_hH1,NVS_GPCR_hLTB4_BLT1,NVS_GPCR_hM1,NVS_GPCR_hM2,NVS_GPCR_hM3,NVS_GPCR_hM4,NVS_GPCR_hNK2,NVS_GPCR_hOpiate_D1,NVS_GPCR_hOpiate_mu,NVS_GPCR_hTXA2,NVS_GPCR_p5HT2C,NVS_GPCR_r5HT1_NonSelective,NVS_GPCR_r5HT_NonSelective,NVS_GPCR_rAdra1B,NVS_GPCR_rAdra1_NonSelective,NVS_GPCR_rAdra2_NonSelective,NVS_GPCR_rAdrb_NonSelective,NVS_GPCR_rNK1,NVS_GPCR_rNK3,NVS_GPCR_rOpiate_NonSelective,NVS_GPCR_rOpiate_NonSelectiveNa,NVS_GPCR_rSST,NVS_GPCR_rTRH,NVS_GPCR_rV1,NVS_GPCR_rabPAF,NVS_GPCR_rmAdra2B,NVS_IC_hKhERGCh,NVS_IC_rCaBTZCHL,NVS_IC_rCaDHPRCh_L,NVS_IC_rNaCh_site2,NVS_LGIC_bGABARa1,NVS_LGIC_h5HT3,NVS_LGIC_hNNR_NBungSens,NVS_LGIC_rGABAR_NonSelective,NVS_LGIC_rNNR_BungSens,NVS_MP_hPBR,NVS_MP_rPBR,NVS_NR_bER,NVS_NR_bPR,NVS_NR_cAR,NVS_NR_hAR,NVS_NR_hCAR_Antagonist,NVS_NR_hER,NVS_NR_hFXR_Agonist,NVS_NR_hFXR_Antagonist,NVS_NR_hGR,NVS_NR_hPPARa,NVS_NR_hPPARg,NVS_NR_hPR,NVS_NR_hPXR,NVS_NR_hRAR_Antagonist,NVS_NR_hRARa_Agonist,NVS_NR_hTRa_Antagonist,NVS_NR_mERa,NVS_NR_rAR,NVS_NR_rMR,NVS_OR_gSIGMA_NonSelective,NVS_TR_gDAT,NVS_TR_hAdoT,NVS_TR_hDAT,NVS_TR_hNET,NVS_TR_hSERT,NVS_TR_rNET,NVS_TR_rSERT,NVS_TR_rVMAT2,OT_AR_ARELUC_AG_1440,OT_AR_ARSRC1_0480,OT_AR_ARSRC1_0960,OT_ER_ERaERa_0480,OT_ER_ERaERa_1440,OT_ER_ERaERb_0480,OT_ER_ERaERb_1440,OT_ER_ERbERb_0480,OT_ER_ERbERb_1440,OT_ERa_EREGFP_0120,OT_ERa_EREGFP_0480,OT_FXR_FXRSRC1_0480,OT_FXR_FXRSRC1_1440,OT_NURR1_NURR1RXRa_0480,OT_NURR1_NURR1RXRa_1440,TOX21_ARE_BLA_Agonist_ch1,TOX21_ARE_BLA_Agonist_ch2,TOX21_ARE_BLA_agonist_ratio,TOX21_ARE_BLA_agonist_viability,TOX21_AR_BLA_Agonist_ch1,TOX21_AR_BLA_Agonist_ch2,TOX21_AR_BLA_Agonist_ratio,TOX21_AR_BLA_Antagonist_ch1,TOX21_AR_BLA_Antagonist_ch2,TOX21_AR_BLA_Antagonist_ratio,TOX21_AR_BLA_Antagonist_viability,TOX21_AR_LUC_MDAKB2_Agonist,TOX21_AR_LUC_MDAKB2_Antagonist,TOX21_AR_LUC_MDAKB2_Antagonist2,TOX21_AhR_LUC_Agonist,TOX21_Aromatase_Inhibition,TOX21_AutoFluor_HEK293_Cell_blue,TOX21_AutoFluor_HEK293_Media_blue,TOX21_AutoFluor_HEPG2_Cell_blue,TOX21_AutoFluor_HEPG2_Cell_green,TOX21_AutoFluor_HEPG2_Media_blue,TOX21_AutoFluor_HEPG2_Media_green,TOX21_ELG1_LUC_Agonist,TOX21_ERa_BLA_Agonist_ch1,TOX21_ERa_BLA_Agonist_ch2,TOX21_ERa_BLA_Agonist_ratio,TOX21_ERa_BLA_Antagonist_ch1,TOX21_ERa_BLA_Antagonist_ch2,TOX21_ERa_BLA_Antagonist_ratio,TOX21_ERa_BLA_Antagonist_viability,TOX21_ERa_LUC_BG1_Agonist,TOX21_ERa_LUC_BG1_Antagonist,TOX21_ESRE_BLA_ch1,TOX21_ESRE_BLA_ch2,TOX21_ESRE_BLA_ratio,TOX21_ESRE_BLA_viability,TOX21_FXR_BLA_Antagonist_ch1,TOX21_FXR_BLA_Antagonist_ch2,TOX21_FXR_BLA_agonist_ch2,TOX21_FXR_BLA_agonist_ratio,TOX21_FXR_BLA_antagonist_ratio,TOX21_FXR_BLA_antagonist_viability,TOX21_GR_BLA_Agonist_ch1,TOX21_GR_BLA_Agonist_ch2,TOX21_GR_BLA_Agonist_ratio,TOX21_GR_BLA_Antagonist_ch2,TOX21_GR_BLA_Antagonist_ratio,TOX21_GR_BLA_Antagonist_viability,TOX21_HSE_BLA_agonist_ch1,TOX21_HSE_BLA_agonist_ch2,TOX21_HSE_BLA_agonist_ratio,TOX21_HSE_BLA_agonist_viability,TOX21_MMP_ratio_down,TOX21_MMP_ratio_up,TOX21_MMP_viability,TOX21_NFkB_BLA_agonist_ch1,TOX21_NFkB_BLA_agonist_ch2,TOX21_NFkB_BLA_agonist_ratio,TOX21_NFkB_BLA_agonist_viability,TOX21_PPARd_BLA_Agonist_viability,TOX21_PPARd_BLA_Antagonist_ch1,TOX21_PPARd_BLA_agonist_ch1,TOX21_PPARd_BLA_agonist_ch2,TOX21_PPARd_BLA_agonist_ratio,TOX21_PPARd_BLA_antagonist_ratio,TOX21_PPARd_BLA_antagonist_viability,TOX21_PPARg_BLA_Agonist_ch1,TOX21_PPARg_BLA_Agonist_ch2,TOX21_PPARg_BLA_Agonist_ratio,TOX21_PPARg_BLA_Antagonist_ch1,TOX21_PPARg_BLA_antagonist_ratio,TOX21_PPARg_BLA_antagonist_viability,TOX21_TR_LUC_GH3_Agonist,TOX21_TR_LUC_GH3_Antagonist,TOX21_VDR_BLA_Agonist_viability,TOX21_VDR_BLA_Antagonist_ch1,TOX21_VDR_BLA_agonist_ch2,TOX21_VDR_BLA_agonist_ratio,TOX21_VDR_BLA_antagonist_ratio,TOX21_VDR_BLA_antagonist_viability,TOX21_p53_BLA_p1_ch1,TOX21_p53_BLA_p1_ch2,TOX21_p53_BLA_p1_ratio,TOX21_p53_BLA_p1_viability,TOX21_p53_BLA_p2_ch1,TOX21_p53_BLA_p2_ch2,TOX21_p53_BLA_p2_ratio,TOX21_p53_BLA_p2_viability,TOX21_p53_BLA_p3_ch1,TOX21_p53_BLA_p3_ch2,TOX21_p53_BLA_p3_ratio,TOX21_p53_BLA_p3_viability,TOX21_p53_BLA_p4_ch1,TOX21_p53_BLA_p4_ch2,TOX21_p53_BLA_p4_ratio,TOX21_p53_BLA_p4_viability,TOX21_p53_BLA_p5_ch1,TOX21_p53_BLA_p5_ch2,TOX21_p53_BLA_p5_ratio,TOX21_p53_BLA_p5_viability,Tanguay_ZF_120hpf_AXIS_up,Tanguay_ZF_120hpf_ActivityScore,Tanguay_ZF_120hpf_BRAI_up,Tanguay_ZF_120hpf_CFIN_up,Tanguay_ZF_120hpf_CIRC_up,Tanguay_ZF_120hpf_EYE_up,Tanguay_ZF_120hpf_JAW_up,Tanguay_ZF_120hpf_MORT_up,Tanguay_ZF_120hpf_OTIC_up,Tanguay_ZF_120hpf_PE_up,Tanguay_ZF_120hpf_PFIN_up,Tanguay_ZF_120hpf_PIG_up,Tanguay_ZF_120hpf_SNOU_up,Tanguay_ZF_120hpf_SOMI_up,Tanguay_ZF_120hpf_SWIM_up,Tanguay_ZF_120hpf_TRUN_up,Tanguay_ZF_120hpf_TR_up,Tanguay_ZF_120hpf_YSE_up
    #     ]

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
        dataset = MolTestDatasetWrapper(batch_size=batch_size, num_workers=4, valid_size=0.1, test_size=0.1, data_path=path, target=target,task=task, splitting='scaffold')
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
