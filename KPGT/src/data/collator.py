import dgl
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .featurizer import smiles_to_graph

def preprocess_batch_light(batch_num, batch_num_target, tensor_data):
    batch_num = np.concatenate([[0],batch_num],axis=-1)
    cs_num = np.cumsum(batch_num)
    add_factors = np.concatenate([[cs_num[i]]*batch_num_target[i] for i in range(len(cs_num)-1)], axis=-1)
    return tensor_data + torch.from_numpy(add_factors).reshape(-1,1)

class Collator_pretrain(object):
    def __init__(
        self, 
        vocab, 
        max_length, n_virtual_nodes, add_self_loop=True,
        candi_rate=0.15, mask_rate=0.8, replace_rate=0.1, keep_rate=0.1,
        fp_disturb_rate=0.15, md_disturb_rate=0.15, 
        data_aug1=None, data_aug1_rate=0.2, data_aug2=None, data_aug2_rate=0.2
        ):
        self.vocab = vocab
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

        self.candi_rate = candi_rate
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.keep_rate = keep_rate

        self.fp_disturb_rate = fp_disturb_rate
        self.md_disturb_rate = md_disturb_rate
        
        self.data_aug1 = data_aug1
        self.data_aug1_rate = data_aug1_rate
        self.data_aug2 = data_aug2
        self.data_aug2_rate = data_aug2_rate
        
    def bert_mask_nodes(self, g):
        n_nodes = g.num_nodes()
        all_ids = np.arange(0, n_nodes, 1, dtype=np.int64)
        
        valid_mask_nodes = torch.where(g.ndata['contrastive_mark'] == 0)[0].numpy()
        valid_ids = valid_mask_nodes[g.ndata['vavn'][valid_mask_nodes] <= 0]
        # valid_ids = torch.where(g.ndata['vavn']<=0)[0].numpy()
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
    
    def disturb_fp(self, fp):
        fp = deepcopy(fp)
        b, d = fp.shape
        fp = fp.reshape(-1)
        disturb_ids = np.random.choice(b*d, int(b*d*self.fp_disturb_rate), replace=False)
        fp[disturb_ids] = 1 - fp[disturb_ids]
        
        return fp.reshape(b,d)
    
    def disturb_md(self, md):
        md = deepcopy(md)
        b, d = md.shape
        md = md.reshape(-1)
        sampled_ids = np.random.choice(b*d, int(b*d*self.md_disturb_rate), replace=False)
        a = torch.empty(len(sampled_ids)).uniform_(0, 1)
        sampled_md = a
        md[sampled_ids] = sampled_md
        
        return md.reshape(b,d)
    
    def data_augment(self, augment, aug_ratio, params):
        edges, atom_pairs_features_in_triplets, bond_features_in_triplets, triplet_labels, virtual_atom_and_virtual_node_labels, paths, line_graph_path_labels, mol_graph_path_labels, virtual_path_labels, self_loop_labels = params
        
        if augment == None:
            pass
        
        elif augment == 'drop_nodes':
            node_num = max(edges[:,0]) + 1
            edge_num = edges.shape[0]
            drop_num = int(node_num * aug_ratio)
            idx_perm = np.random.permutation(node_num - self.n_virtual_nodes)
            
            idx_drop = idx_perm[:drop_num].tolist()
            idx_nondrop = idx_perm[drop_num:].tolist()
            idx_nondrop.extend(list(range(node_num))[-self.n_virtual_nodes:])
                            
            idx_nondrop.sort()
            idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
            
            # nodes drop
            atom_pairs_features_in_triplets = atom_pairs_features_in_triplets[idx_nondrop]
            bond_features_in_triplets = bond_features_in_triplets[idx_nondrop]
            triplet_labels = triplet_labels[idx_nondrop]
            virtual_atom_and_virtual_node_labels = virtual_atom_and_virtual_node_labels[idx_nondrop]
            
            # corresponding edges drop
            for node in idx_drop:
                paths = torch.where((paths == node), idx_nondrop[-self.n_virtual_nodes], paths)

            edge_nondrop = np.array([n for n in range(edge_num) if not (edges[n, 0] in idx_drop or edges[n, 1] in idx_drop)]) # edge_num x 2
            edges = edges[edge_nondrop]
            paths = paths[edge_nondrop]
            line_graph_path_labels = line_graph_path_labels[edge_nondrop]
            mol_graph_path_labels = mol_graph_path_labels[edge_nondrop]
            virtual_path_labels = virtual_path_labels[edge_nondrop]
            self_loop_labels = self_loop_labels[edge_nondrop]

            for key in idx_nondrop:
                edges = np.where((edges == key), idx_dict[key], edges)
                paths = torch.where((paths == key), idx_dict[key], paths)
        
        elif augment == 'permute_edges':
            node_num = max(edges[:,0]) + 1
            edge_num = edges.shape[0]
            permute_num = int(edge_num * aug_ratio)
            idx_add = np.random.choice(node_num, (permute_num, 2))
            
            edge_nondrop = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
            edge_drop = np.array([n for n in range(edge_num) if not n in edge_nondrop])
            edge_index = np.concatenate((edges[edge_nondrop, :], idx_add), axis=0)
            
            edges = edge_index
            paths = torch.cat((paths[edge_nondrop], paths[edge_drop]), dim=0)
            line_graph_path_labels = torch.cat((line_graph_path_labels[edge_nondrop], line_graph_path_labels[edge_drop]), dim=0)
            mol_graph_path_labels = torch.cat((mol_graph_path_labels[edge_nondrop], mol_graph_path_labels[edge_drop]), dim=0)
            virtual_path_labels = torch.cat((virtual_path_labels[edge_nondrop], virtual_path_labels[edge_drop]), dim=0)
            self_loop_labels = torch.cat((self_loop_labels[edge_nondrop], self_loop_labels[edge_drop]), dim=0)
            
        elif augment == 'mask_nodes':
            node_num = max(edges[:,0]) + 1
            mask_num = int(node_num * aug_ratio)
            idx_mask = np.random.choice(node_num, mask_num, replace=False)
            
            begin_end_token = atom_pairs_features_in_triplets.mean(dim=0)
            edge_token = bond_features_in_triplets.mean(dim=0)
            
            atom_pairs_features_in_triplets[idx_mask] = begin_end_token
            bond_features_in_triplets[idx_mask] = edge_token
        
        elif augment == 'subgraph':
            node_num = max(edges[:,0]) + 1
            edge_num = edges.shape[0]
            sub_num = int(node_num * aug_ratio)
            
            virtual = torch.nonzero(virtual_atom_and_virtual_node_labels).view(-1).tolist()
            idx_sub = [np.random.randint(node_num, size=1)[0]] # random walk start node

            edge_index = edges.T # 2*edge_num
            idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])
            
            count = 0
            while len(idx_sub) <= sub_num:
                count = count + 1
                if count > node_num:
                    break
                if len(idx_neigh) == 0:
                    break
                sample_node = np.random.choice(list(idx_neigh)) # choose one node from start node's neighbor
                if sample_node in idx_sub or virtual_atom_and_virtual_node_labels[sample_node] != 0: # repetitive sample & sample virtual nodes
                    continue
                idx_sub.append(sample_node)
                idx_neigh.union(set([n for n in edges[1][edges[0]==idx_sub[-1]]])) # continue random sampling
            
            idx_drop = [n for n in range(node_num) if not n in idx_sub]
            idx_nondrop = idx_sub
            for n in virtual:
                if n not in idx_nondrop:
                    idx_nondrop.append(n)
                    idx_drop.remove(n)

            idx_nondrop.sort()
            idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

            # nodes drop
            atom_pairs_features_in_triplets = atom_pairs_features_in_triplets[idx_nondrop]
            bond_features_in_triplets = bond_features_in_triplets[idx_nondrop]
            triplet_labels = triplet_labels[idx_nondrop]
            virtual_atom_and_virtual_node_labels = virtual_atom_and_virtual_node_labels[idx_nondrop]
            
            # corresponding edges drop
            for node in idx_drop:
                paths = torch.where((paths == node), idx_nondrop[-self.n_virtual_nodes], paths)

            edge_nondrop = np.array([n for n in range(edge_num) if not (edges[n, 0] in idx_drop or edges[n, 1] in idx_drop)]) # edge_num x 2
            edges = edges[edge_nondrop]
            paths = paths[edge_nondrop]
            line_graph_path_labels = line_graph_path_labels[edge_nondrop]
            mol_graph_path_labels = mol_graph_path_labels[edge_nondrop]
            virtual_path_labels = virtual_path_labels[edge_nondrop]
            self_loop_labels = self_loop_labels[edge_nondrop]

            for key in idx_nondrop:
                edges = np.where((edges == key), idx_dict[key], edges)
                paths = torch.where((paths == key), idx_dict[key], paths)
        
        else:
            raise ValueError('Unknown data augmentation!')
        
        data = (edges[:,0], edges[:,1])
        g = dgl.graph(data)
        g.ndata['begin_end'] = atom_pairs_features_in_triplets
        g.ndata['edge'] = bond_features_in_triplets
        g.ndata['label'] = triplet_labels
        g.ndata['vavn'] = virtual_atom_and_virtual_node_labels
        g.edata['path'] = paths
        g.edata['lgp'] = line_graph_path_labels
        g.edata['mgp'] = mol_graph_path_labels
        g.edata['vp'] = virtual_path_labels
        g.edata['sl'] = self_loop_labels
        return g
    
    def __call__(self, samples):
        smiles_list, fps, mds = map(list, zip(*samples))
        graphs = []
        contrastive_graph1 = []
        contrastive_graph2 = []
        for smiles in smiles_list:
            params = smiles_to_graph(smiles, self.vocab, max_length=self.max_length, n_virtual_nodes=self.n_virtual_nodes, add_self_loop=self.add_self_loop)
            graph0 = self.data_augment(augment=None, aug_ratio=None, params=params)
            graph1 = self.data_augment(augment=self.data_aug1, aug_ratio=self.data_aug1_rate, params=params)
            graph2 = self.data_augment(augment=self.data_aug2, aug_ratio=self.data_aug2_rate, params=params)
            graph0.ndata['contrastive_mark'] = torch.zeros(graph0.num_nodes()) # 0 denotes original graph
            graph1.ndata['contrastive_mark'] = torch.ones(graph1.num_nodes()) # 1 denotes data augmentation method 1
            graph2.ndata['contrastive_mark'] = torch.ones(graph2.num_nodes()) * 2 # 2 denotes data augmentation method 2
            graphs.append(graph0)
            contrastive_graph1.append(graph1)
            contrastive_graph2.append(graph2)
        graphs.extend(contrastive_graph1)
        graphs.extend(contrastive_graph2)
        batched_graph = dgl.batch(graphs)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list),-1)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list),-1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(), batched_graph.batch_num_edges(), batched_graph.edata['path'][:, :])
        sl_labels = self.bert_mask_nodes(batched_graph)
        disturbed_fps = self.disturb_fp(fps)
        disturbed_mds = self.disturb_md(mds)
        
        # similarly, the fps and mds should be added twice
        disturbed_mds = torch.cat([disturbed_mds, mds, mds], dim=0)
        disturbed_fps = torch.cat([disturbed_fps, fps, fps], dim=0)
        fps = torch.cat([fps, fps, fps], dim=0)
        mds = torch.cat([mds, mds, mds], dim=0)

        return smiles_list, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds

class Collator_tune(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop
    def __call__(self, samples):
        smiles_list, graphs, fps, mds, labels = map(list, zip(*samples))

        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list),-1)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list),-1)
        labels = torch.stack(labels, dim=0).reshape(len(smiles_list),-1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(), batched_graph.batch_num_edges(), batched_graph.edata['path'][:, :])
        return smiles_list, batched_graph, fps, mds, labels
