import csv
import numpy as np

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.loader import DataLoader
from typing import List 

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task

        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        data = HeteroData()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        data['atom'].x = x


        bond_types = {
            'SINGLE': {'row': [], 'col': [], 'features': []},
            'DOUBLE': {'row': [], 'col': [], 'features': []},
            'TRIPLE': {'row': [], 'col': [], 'features': []},
            'AROMATIC': {'row': [], 'col': [], 'features': []}
        }

        def add_bond(bond_type, start, end, features):
            bond_types[bond_type]['row'] += [start, end]
            bond_types[bond_type]['col'] += [end, start]
            bond_types[bond_type]['features'].append(features)
            bond_types[bond_type]['features'].append(features)

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            features = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
            bond_type = str(bond.GetBondType())
            if bond_type in bond_types:
                add_bond(bond_type, start, end, features)
            else:
                print('unknown edge type!')

        def process_edge(bond_type):
            idx = torch.tensor([bond_types[bond_type]['row'], bond_types[bond_type]['col']], dtype=torch.long)
            attr = torch.tensor(np.array(bond_types[bond_type]['features']), dtype=torch.long)
        
            return idx, attr
        
        edge_index_single, edge_attr_single = process_edge('SINGLE')
        edge_index_double, edge_attr_double = process_edge('DOUBLE')
        edge_index_triple, edge_attr_triple = process_edge('TRIPLE')
        edge_index_aromatic, edge_attr_aromatic = process_edge('AROMATIC')

        data['atom', 'SINGLE', 'atom'].edge_index = edge_index_single
        data['atom', 'DOUBLE', 'atom'].edge_index = edge_index_double
        data['atom', 'TRIPLE', 'atom'].edge_index = edge_index_triple
        data['atom', 'AROMATIC', 'atom'].edge_index = edge_index_aromatic
        data['atom', 'SINGLE', 'atom'].edge_attr = edge_attr_single
        data['atom', 'DOUBLE', 'atom'].edge_attr = edge_attr_double
        data['atom', 'TRIPLE', 'atom'].edge_attr = edge_attr_triple
        data['atom', 'AROMATIC', 'atom'].edge_attr = edge_attr_aromatic

        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
        
        data['atom'].y = y
        return data

    def __len__(self):
        return len(self.smiles_data)

    def get(self):
        pass

    def len(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_path, target, task, splitting
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader
