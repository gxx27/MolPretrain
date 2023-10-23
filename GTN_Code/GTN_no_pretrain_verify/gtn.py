import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from gcn import GCNConv
from torch_scatter import scatter_add
from torch_geometric.nn import global_mean_pool
import torch_sparse

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 


class GTN(nn.Module):   
    def __init__(self, task ,num_channels, w_in, w_out, num_layers, emb_dim, args=None):
        super(GTN, self).__init__()
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_layers = num_layers
        self.args = args
        self.task = task

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_bond_type, num_channels, first=True))
            else:
                layers.append(GTLayer(num_bond_type, num_channels, first=False))
        self.layers = nn.ModuleList(layers)

        self.gcn = GCNConv(in_channels=self.w_in, out_channels=w_out, args=args)
        self.pool = global_mean_pool
        self.linear = nn.Linear(self.w_out*self.num_channels, self.w_out)       

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        self.edge_embedding1 = nn.Embedding(num_bond_type, 1)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, 1)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def normalization(self, H, num_nodes):
        norm_H = []
        for i in range(self.num_channels):
            edge, value=H[i]
            deg_row, deg_col = self.norm(edge.detach(), num_nodes, value)
            value = (deg_row) * value
            norm_H.append((edge, value))
        return norm_H

    def norm(self, edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                    dtype=dtype,
                                    device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        row, col = edge_index
        deg = scatter_add(edge_weight.clone(), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    def forward(self, data, num_nodes=None, eval=False):
        X = data['atom'].x
        num_nodes = X.shape[0]
        h = self.x_embedding1(X[:,0]) + self.x_embedding2(X[:,1])
        h = 0.1*h + 0.9*h.detach()

        A = []

        # SINGLE
        single_index = data['SINGLE']['edge_index']
        single_attr = data['SINGLE']['edge_attr']
        if len(single_attr) != 0:
            attr_single = self.edge_embedding1(single_attr[:,0]) + self.edge_embedding2(single_attr[:,1])
            # attr_single = single_attr[:,1]
            A.append((single_index, attr_single.squeeze(-1)))

        # DOUBLE
        double_index = data['DOUBLE']['edge_index']
        double_attr = data['DOUBLE']['edge_attr']
        if len(double_attr) != 0:
            attr_double = self.edge_embedding1(double_attr[:,0]) + self.edge_embedding2(double_attr[:,1])
            # attr_double = double_attr[:,1]
            A.append((double_index, attr_double.squeeze(-1)))

        # TRIPLE
        triple_index = data['TRIPLE']['edge_index']
        triple_attr = data['TRIPLE']['edge_attr']
        if len(triple_attr) != 0:
            attr_triple = self.edge_embedding1(triple_attr[:,0]) + self.edge_embedding2(triple_attr[:,1])
            # attr_triple = triple_attr[:,1]
            A.append((triple_index, attr_triple.squeeze(-1)))

        # AROMATIC
        aromatic_index = data['AROMATIC']['edge_index']
        aromatic_attr = data['AROMATIC']['edge_attr']
        if len(aromatic_attr) != 0:
            attr_aromatic = self.edge_embedding1(aromatic_attr[:,0]) + self.edge_embedding2(aromatic_attr[:,1])   
            # attr_aromatic = aromatic_attr[:,1]
            A.append((aromatic_index, attr_aromatic.squeeze(-1)))

        # self-loop
        edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).to(X.device)
        value_tmp = torch.ones(num_nodes,dtype=torch.float32).to(X.device)
        A.append((edge_tmp,value_tmp))

        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A, num_nodes, eval=eval)
            else:                
                H, W = self.layers[i](A, num_nodes, H, eval=eval)
            H = self.normalization(H, num_nodes)
            Ws.append(W)

        for i in range(self.num_channels):
            edge_index, edge_weight = H[i][0], H[i][1]
            if i==0:
                # h = self.batch_norms[i](h)
                X_ = self.gcn(h,edge_index=edge_index.detach(), edge_weight=edge_weight)
                # X_ = self.batch_norms[i](X_)
                X_ = F.relu(X_)
            else:
                # h = self.batch_norms[i](h)
                X_tmp = self.gcn(h,edge_index=edge_index.detach(), edge_weight=edge_weight)
                # X_tmp = self.batch_norms[i](X_tmp)
                X_tmp = F.relu(X_tmp)
                X_ = torch.cat((X_,X_tmp), dim=1)

        y = self.pool(X_, data['atom'].batch)

        return self.linear(y)
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

class GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first

        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels) # Q1
            self.conv2 = GTConv(in_channels, out_channels) # Q2
        else:
            self.conv1 = GTConv(in_channels, out_channels) # Q3
    
    def forward(self, A, num_nodes, H_=None, eval=False):
        if self.first == True:
            result_A = self.conv1(A, num_nodes, eval=eval)
            result_B = self.conv2(A, num_nodes, eval=eval)
            W = [(F.softmax(self.conv1.weight, dim=1)),(F.softmax(self.conv2.weight, dim=1))]
        else:
            result_A = H_
            result_B = self.conv1(A, num_nodes, eval=eval)
            W = [(F.softmax(self.conv1.weight, dim=1))]
            
        H = []
        for i in range(len(result_A)): 
            a_edge, a_value = result_A[i]
            b_edge, b_value = result_B[i]
            mat_a = torch.sparse_coo_tensor(a_edge, a_value, (num_nodes, num_nodes)).to(a_edge.device)
            mat_b = torch.sparse_coo_tensor(b_edge, b_value, (num_nodes, num_nodes)).to(a_edge.device)
            mat = torch.sparse.mm(mat_a, mat_b).coalesce()
            edges, values = mat.indices(), mat.values()

            H.append((edges, values))
        return H, W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels))
        self.bias = None
        self.reset_parameters()
        
    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, std=0.01)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A, num_nodes, eval=eval):
        filter = F.softmax(self.weight, dim=1)
        num_channels = self.weight.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index,edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value*filter[i][j]
                    # total_edge_value = edge_value
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value*filter[i][j]))
                    # total_edge_value = torch.cat((total_edge_value, edge_value))

            index, value = torch_sparse.coalesce(total_edge_index.detach().type(torch.long), total_edge_value, m=num_nodes, n=num_nodes, op='add')
            results.append((index, value))

        return results