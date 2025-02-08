# coding=utf-8
import os
import json
import re
import argparse
import math
import networkx as nx
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import KFold
from torch_scatter import scatter_softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    Size,
    OptPairTensor,
    PairTensor,
    NoneType
)
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)

from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import SparseTensor, masked_select_nnz, matmul
from torch_geometric.nn.inits import glorot, zeros, ones
from torch_geometric.utils import softmax,is_torch_sparse_tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, BertForMaskedLM
from typing import Optional, Tuple, Union

from test import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("./models--CSHaitao--SAILER_zh")
sailer_model = AutoModel.from_pretrained("./models--CSHaitao--SAILER_zh").to(device)

parser = argparse.ArgumentParser(description = 'lcr')
parser.add_argument('--dataset', type = str, default = 'lecardv2', 
                    help = 'dataset_name')  
parser.add_argument('--model', type = str, default = 'EAGATv2-EFWF', 
                    help = 'model_name')  
parser.add_argument('--fold', type = str, default = 0,
                    help = 'indicate fold for lecard (0,1)')
parser.add_argument('--exp_name', type = str, default = 'exp_1', 
                    help = 'experiment_name')  
parser.add_argument('--drop_mode', type = str, default = False, 
                    help = 'whether to adopt drop mode')
parser.add_argument('--drop_type', type = str, default = 'most', 
                    help = 'indicate drop type (few or most)')
parser.add_argument('--attribute_file', type = str, default = None, 
                    help = 'attribute filtering result')
parser.add_argument('--mode', type = str, default = 'test', 
                    help = 'Specifies the mode of operation (train, test)')
parser.add_argument('--checkpoint', type = str, default = './checkpoints/only_pe_no_droptrain_val.pth', 
                    help = 'load the checkpoint for testing')
parser.add_argument('--retrain_times', type = str, default = 10, 
                    help = 'training times')
parser.add_argument('--epochs', type = str, default = 10, 
                    help = 'epochs')

args = parser.parse_args()

def encode_text(text, length=64):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=length).to(device)
        outputs = sailer_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)

def save_to_pkl(data_list, file_path):
    with open(file_path, 'wb') as f:
        pkl.dump(data_list, f)
    print(f"Data saved to {file_path}")

def collate_fn(batch):
    return batch

def load_data_kfold(datas, k_folds=5, batch_size=1, part_few = None):
    case_dataset = CaseDataset(datas, dataset = 'lecard', part_few = part_few)
    dataset_size = len(case_dataset)
    kf = KFold(n_splits=k_folds, shuffle=False)
    kfold_loaders = []  # 存储每一折的 DataLoader

    for train_indices, val_indices in kf.split(range(dataset_size)):
        train_dataset = Subset(case_dataset, train_indices)
        val_dataset = Subset(case_dataset, val_indices)
        train_loader = DataLoader(train_dataset, batch_size = batch_size, collate_fn=collate_fn, shuffle= True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

        kfold_loaders.append({'train': train_loader, 'val': val_loader})

    return kfold_loaders

def load_data(datas,train_test, part_few = None):
    train_ratio = 0.8
    case_dataset = CaseDataset(datas, dataset = 'lecardv2', part_few = part_few)
    train_indices = []
    val_indices = []
    for idx, case_batch in enumerate(case_dataset.cached_data):
        case_number = case_batch.case_number 
        if int(case_number[0]) in train_test['test']:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    train_dataset = Subset(case_dataset, train_indices)
    val_dataset = Subset(case_dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

    return train_loader,val_loader

class EdgeAwareGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, num_heads=1, 
                 dropout=0.0, bias=True, add_self_loops=False):
        super(EdgeAwareGATConv, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.num_heads = num_heads
        self.add_self_loops = add_self_loops
        
        self.fc = nn.Linear(in_channels, num_heads * out_channels, bias=False)
        self.fc_edge = nn.Linear(edge_channels, num_heads * out_channels, bias=False)
        
        self.attn_l = nn.Parameter(torch.Tensor(1, num_heads, out_channels))
        self.attn_r = nn.Parameter(torch.Tensor(1, num_heads, out_channels))
        self.attn_edge = nn.Parameter(torch.Tensor(1, num_heads, out_channels))
        
        self.res_fc = nn.Linear(in_channels, num_heads * out_channels, bias=False) if bias else None
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.flow = 'target_to_source'
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.fc_edge.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        nn.init.xavier_uniform_(self.attn_edge)
        if self.res_fc is not None:
            nn.init.xavier_uniform_(self.res_fc.weight)

    def forward(self, x, edge_index, edge_attr):
        x_transformed = self.fc(x).view(-1, self.num_heads, self.out_channels)
        edge_transformed = self.fc_edge(edge_attr).view(-1, self.num_heads, self.out_channels)
        
        alpha_l = (x_transformed * self.attn_l).sum(dim=-1)
        alpha_r = (x_transformed * self.attn_r).sum(dim=-1)
        alpha_e = (edge_transformed * self.attn_edge).sum(dim=-1)
        
        alpha = alpha_l[edge_index[0]] + alpha_r[edge_index[1]] + alpha_e
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index[1])
        alpha = self.dropout(alpha)
        updated_features = x_transformed.clone() 
        out = self.propagate(edge_index, x=x_transformed, edge_attr=edge_transformed, alpha=alpha)
        updated_features[:len(out)] += out 
        if self.res_fc is not None:
            res = self.res_fc(x).view(-1, self.num_heads, self.out_channels)
            updated_features += res
        
        return updated_features.squeeze(1)
    def message(self, x_j, edge_attr, alpha):
        return (x_j + edge_attr) * alpha.view(-1, self.num_heads, 1)
    
    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=0, reduce='sum')


class CaseGNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout, num_heads):
        super(CaseGNN, self).__init__()
        self.hidden_size = h_dim
        # 使用 PyTorch Geometric 的 GATConv
        self.EdgeGATConv1 = EdgeAwareGATConv(in_dim, in_dim, in_dim, num_heads)
        self.EdgeGATConv2 = EdgeAwareGATConv(in_dim, in_dim, out_dim, num_heads)
        
        self.embedding_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        if self.hidden_size == 0:
            stdv = 1.0 / math.sqrt(self.in_dim)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)
        else:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)


    def forward(self, node_feats,edge_index,edge_feats):
        h = self.EdgeGATConv1(node_feats, edge_index, edge_feats)
        h = self.embedding_dropout(h)

        if self.hidden_size == 0:
            pool = global_mean_pool(h, data.batch)
            return pool
        else:
            h = F.relu(h)
            h = self.EdgeGATConv2(h, edge_index, edge_feats)
            h = self.dropout(h)
            return h
from collections import Counter
def count_edge_types(edge_type):
    edge_counter = Counter(edge_type)  # 统计每种边的频率
    return edge_counter
def compute_edge_weights(edge_type_list):
    # 统计每种边的频率
    edge_counter = Counter(edge_type_list)
    
    # 计算总边数
    total_edges = sum(edge_counter.values())

    # 计算权重：逆频率（IDF-like）
    edge_weights_dict = {etype: count for etype, count in edge_counter.items()}

    # 归一化权重（Softmax）
    weight_values = torch.tensor(list(edge_weights_dict.values()), dtype=torch.float32)
    # print(weight_values)
    normalized_weights = torch.sigmoid(weight_values)

    # 映射回字典
    edge_weights_dict = {etype: normalized_weights[i].item() for i, etype in enumerate(edge_counter.keys())}

    # 生成每条边的权重
    edge_weights = torch.tensor([edge_weights_dict[etype] for etype in edge_type_list], dtype=torch.float32)

    return edge_weights
class EAGATv2_EFWF(MessagePassing):
    r"""The EAGATv2-EFWF operator is an extension of the GATv2 operator from the "How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>_ paper, which addresses the static attention issue in the
    standard :class:~torch_geometric.conv.GATConv layer. EAGATv2-EFWF enhances the GATv2 framework by introducing
    an Edge Feature Weighting Fusion (EFWF) mechanism that dynamically adjusts the edge weights based on the
    feature representations of edges. 

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:-1 to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities in case of a bipartite graph.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:1)
        concat (bool, optional): If set to :obj:False, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:True)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:0.2)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:0)
        add_self_loops (bool, optional): If set to :obj:False, will not add
            self-loops to the input graph. (default: :obj:True)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:None)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:edge_dim != None).
            If given as :obj:float or :class:torch.Tensor, edge features of
            self-loops will be directly given by :obj:fill_value.
            If given as :obj:str, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:"add", :obj:"mean",
            :obj:"min", :obj:"max", :obj:"mul"). (default: :obj:"mean")
        bias (bool, optional): If set to :obj:False, the layer will not learn
            an additive bias. (default: :obj:True)
        share_weights (bool, optional): If set to :obj:True, the same matrix
            will be applied to the source and the target node of every edge,
            *i.e.* :math:\mathbf{\Theta}_{s} = \mathbf{\Theta}_{t}.
            (default: :obj:False)
        residual (bool, optional): If set to :obj:True, the layer will add
            a learnable skip-connection. (default: :obj:False)
        **kwargs (optional): Additional arguments of
            :class:torch_geometric.nn.conv.MessagePassing.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        aggr: str = 'sum',
        wg_dim: int = 64,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights
        self.flow = 'target_to_source'

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, out_channels))
        
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.edge_encode = nn.Linear(in_channels*2,in_channels)
        self.weight = nn.Linear(in_channels*2,2)
        self.output_proj = torch.nn.Linear(in_channels, in_channels, bias=True)
        self.weight_generator = torch.nn.Sequential(
            torch.nn.Linear(in_channels,wg_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(wg_dim, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
        self.reset_parameters()
        
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        # self.weight_generator.reset_parameters()
        for layer in self.weight_generator:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        edge_type:  Optional = None,
        return_attention_weights: Optional[bool] = None,
        edge_weight: OptTensor = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2

            if self.res is not None:
                res = self.res(x)

            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2

            if x_r is not None and self.res is not None:
                res = self.res(x_r)

            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        alpha,edge_relation_weight = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr,edge_type=edge_type)
        edge_attr = edge_attr.unsqueeze(1)

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha, edge_attr=edge_attr, edge_relation_weight= edge_relation_weight)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if res is not None:
            out = out + res
        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor, dim_size: Optional[int], edge_type: OptTensor,
                    edge_weight: OptTensor = None) -> Tensor:
        x = x_i + x_j
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            # x = x + edge_attr
           
 

        edge_relation_weight = torch.zeros(len(x)).to(device)
        if edge_type is not None:
            mask = torch.tensor(
                [len(et.split("_")) == 3 for et in edge_type], device=x.device
            )
            selected_edges = torch.nonzero(mask).squeeze(-1)
            # 按 target 结点分组
            unique_targets = torch.unique(index[selected_edges])
            for target in unique_targets:
                mask_target = index[selected_edges] == target
                target_edges = selected_edges[mask_target]
                
                if len(target_edges) > 1:
                    edge_selected = (x)[target_edges].squeeze(1)  # shape: (num_edges, out_channels)
                    # print(x_selected.shape)
                    num_edges = len(target_edges)
                    relation_scores = torch.mm(edge_selected, edge_selected.T)
                    # pairwise_features = torch.cat([
                    #     edge_selected.unsqueeze(1).expand(-1, num_edges, -1),  # (num_edges, num_edges, out_channels)
                    #     edge_selected.unsqueeze(0).expand(num_edges, -1, -1)  # (num_edges, num_edges, out_channels)
                    # ], dim=-1)  # shape: (num_edges, num_edges, 2 * out_channels)
                    # # print(pairwise_features.shape)
                    # relation_scores = self.mlp(pairwise_features)  # shape: (num_edges, num_edges, 1)
                    # relation_scores = relation_scores.squeeze(-1)  # shape: (num_edges, num_edges)
                    # print(relation_scores.shape)
                    relation_weights = relation_scores.sum(dim=-1) / num_edges  # shape: (num_edges,)
                    edge_relation_weight[target_edges] += relation_weights
        from collections import Counter

        edge_weights = compute_edge_weights(edge_type)

        x = F.leaky_relu(x, self.negative_slope)

        alpha = (x * self.att).sum(dim=-1)
        # alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha, edge_weights

    def message(self, x_j: Tensor, alpha: Tensor,edge_attr: Tensor, edge_relation_weight: Tensor) -> Tensor:
        if edge_attr is not None:
            weights = self.weight_generator(x_j+edge_attr)
            weights = F.sigmoid(weights)
            
            x_j_weighted = edge_attr * weights + x_j
        return x_j_weighted * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GATv2Conv(MessagePassing):
    r"""The GATv2 operator from the "How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>_ paper, which fixes the
    static attention problem of the standard
    :class:~torch_geometric.conv.GATConv layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:GATv2, every node can attend to any other node.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:-1 to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities in case of a bipartite graph.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:1)
        concat (bool, optional): If set to :obj:False, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:True)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:0.2)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:0)
        add_self_loops (bool, optional): If set to :obj:False, will not add
            self-loops to the input graph. (default: :obj:True)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:None)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:edge_dim != None).
            If given as :obj:float or :class:torch.Tensor, edge features of
            self-loops will be directly given by :obj:fill_value.
            If given as :obj:str, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:"add", :obj:"mean",
            :obj:"min", :obj:"max", :obj:"mul"). (default: :obj:"mean")
        bias (bool, optional): If set to :obj:False, the layer will not learn
            an additive bias. (default: :obj:True)
        share_weights (bool, optional): If set to :obj:True, the same matrix
            will be applied to the source and the target node of every edge,
            *i.e.* :math:\mathbf{\Theta}_{s} = \mathbf{\Theta}_{t}.
            (default: :obj:False)
        residual (bool, optional): If set to :obj:True, the layer will add
            a learnable skip-connection. (default: :obj:False)
        **kwargs (optional): Additional arguments of
            :class:torch_geometric.nn.conv.MessagePassing.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        aggr: str = 'sum',
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights
        self.flow = 'target_to_source'
        self.ln = nn.Linear(2,1)
        # self.aggr = 'mean'
        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else: 
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, out_channels))
        
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.edge_encode = nn.Linear(in_channels*2,in_channels)
        self.weight = nn.Linear(in_channels*2,2)
        self.output_proj = torch.nn.Linear(in_channels, in_channels, bias=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
        edge_weight: OptTensor = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2

            if self.res is not None:
                res = self.res(x)

            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2

            if x_r is not None and self.res is not None:
                res = self.res(x_r)

            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        alpha,edge_attr1 = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)
        # Propagate messages
        edge_attr = edge_attr.unsqueeze(1)

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res
        # print(self.bias.shape,out.shape)
        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor, dim_size: Optional[int],
                    edge_weight: OptTensor = None,change_weight: OptTensor = None) -> Tensor:
        x = x_i + x_j
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr
            
        
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)

        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha,edge_attr
        

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        edge_types=1,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.type_weight = torch.nn.Parameter(torch.Tensor(edge_types, heads, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)
        glorot(self.type_weight)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,edge_type: OptTensor = None,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")


        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor,index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class CaseDataset(Dataset):
    def __init__(self, data, dataset = 'lecard', part_few = 'None'):
        self.data_list = data
        self.device = torch.device("cuda:0")
        # with open("./dataset/lecardv2_1.pkl",'rb')as f:
        with open('./dataset/lecardv2_in_u_2.pkl', 'rb') as f:
            self.cached_data = pkl.load(f)
        if dataset == 'lecard':
            origin_text_dir = "./dataset/processed_text/lecard/"
        elif dataset == 'lecardv2':
            origin_text_dir = "./dataset/processed_text/lecardv2/"
        with open(origin_text_dir+"query_fact.json","r",encoding='utf-8')as f:
            self.data_query = json.load(f)
            f.close()
        with open(origin_text_dir+"doc_fact.json","r",encoding='utf-8')as f:
            self.data_doc = json.load(f)
            f.close()
        self.text_embedding = {}
        texts = []
        self.fulltext = {}
        for key in self.data_query:
            texts_content = ""
            for crime in self.data_query[key]:
                texts_content += self.data_query[key][crime].split("：")[-1]
            texts.append(texts_content)
            self.fulltext[key] = texts_content
        texts = []
        for key in self.data_doc:
            texts_content = ""
            for crime in self.data_doc[key]:
                texts_content += self.data_doc[key][crime].split("：")[-1]
            texts.append(texts_content)
            self.fulltext[key] = texts_content
        # with open(part_few,"r",encoding='utf-8')as f:
        #     self.part_few = json.load(f)
        #     f.close()
        # self.cached_data = []
        # self.cache_all_data()
    def __len__(self):
        return len(self.data_list)

    def cache_all_data(self):
        for idx, case_batch in enumerate(self.data_list):
            graph_list = [self.build_graph(case) for case in case_batch]
            merged_graph = self.merge_graphs(graph_list)
            data = from_networkx(merged_graph)
            data.x = torch.zeros(len(merged_graph.nodes),768)
            case_mask = torch.tensor([bool(n!= 'crime1') for n in data.node_type])
            data.x[case_mask] = encode_text([merged_graph.nodes[n]["texts"] for n in merged_graph.nodes if merged_graph.nodes[n]["node_type"] != "crime1"]).detach().cpu()
            case_mask = torch.tensor([bool(n == 'crime1') for n in data.node_type]) 
            data.x[case_mask] = encode_text([merged_graph.nodes[n]["texts"] for n in merged_graph.nodes if merged_graph.nodes[n]["node_type"] == "crime1"], length = 300).detach().cpu()
            data.edge_attr = encode_text([merged_graph[u][v]["edge_attr"] for u, v in merged_graph.edges]).detach().cpu()
            data.crime_name = [merged_graph[u][v]["crime_name"] for u, v in merged_graph.edges]
            data.drop = [merged_graph[u][v]["part"] for u, v in merged_graph.edges]
            self.cached_data.append(data)
        # save_to_pkl(self.cached_data, 'lecardv2_in_u_2.pkl')
    def __getitem__(self, idx):
        data = self.cached_data[idx]
        # print(data.edge_type)
        data = data.to(self.device)
        return data

    def build_graph(self, case):
        G = nx.DiGraph()
        default_node_attributes = {}
        change = [0,0,0,0]
        pad_size = 64
        PAD = '<PAD>'
        UNK = '<UNK>'
        case_id = f"case_{case['案件编号']}"
        origin_text = None
        if case['案件编号'] in self.data_doc:
            origin_text = self.data_doc[case['案件编号']]
        elif case['案件编号'] in self.data_query:
            origin_text = self.data_query[case['案件编号']]
        G.add_node(case_id, **default_node_attributes, node_type = "case",case_number = case['案件编号'], name = case_id)
        content = []
        G.nodes[case_id]["texts"] = "None"
        zuiming = []
        for crime_info in case["不同罪名"]:
            crime_name = crime_info["罪名"]
            if crime_name not in zuiming:
                zuiming.append(crime_name)
                crime_node = f"crime_{crime_name}_{case['案件编号']}"
                if crime_name in origin_text:
                    crime_node1 = crime_name
                    if "经过：" in origin_text[crime_name]:
                        G.add_node(crime_node1, **default_node_attributes, node_type = "crime1", case_number = case['案件编号'], name = crime_name + ":" + origin_text[crime_name])
                        G.add_edge(case_id, crime_node1, edge_type = "has_crime1", crime_name = crime_name, part = 0)
                        G.nodes[crime_node1]["texts"] = crime_name+":" + origin_text[crime_name]
                G.add_node(crime_node, **default_node_attributes, node_type = "crime", case_number=case['案件编号'], name = crime_name)
                G.add_edge(case_id, crime_node, edge_type = "has_crime", crime_name = crime_name, part = 0)

                G.nodes[crime_node]["texts"] = crime_name
                for section, content in crime_info.items():
                    section_node = f"{section}_{crime_name}_{case['案件编号']}"
                    if isinstance(content,list):
                        for cont in content:
                            if not str(cont).startswith("未提及") and not str(cont).startswith("未明确"):
                                if self.part_few is not None:
                                    # print(self.part_few)
                                    print(crime_name, section)
                                    if crime_name in self.part_few:
                                        
                                        if f"{section}" in self.part_few[crime_name]:
                                            print("1127**",section_node)
                                            G.add_node(section_node, **default_node_attributes, node_type="attribute", case_number = case['案件编号'], name = cont)
                                            drop = 0
                                            if crime_name in self.part_few:
                                                if f"{section}" in self.part_few[crime_name]:
                                                    drop = 1
                                            if crime_name in self.part_few:
                                                if f"{section}" not in self.part_few[crime_name]:
                                                    drop = 2
                                            G.add_edge(crime_node, section_node, edge_type=f"has_{crime_name}_{section}", crime_name = crime_name, part = drop)
                                            G.nodes[section_node]["texts"] = str(cont)
                                    else:
                                        print("1139**",section_node)
                                        G.add_node(section_node, **default_node_attributes, node_type="attribute", case_number = case['案件编号'], name = cont)
                                        drop = 0
                                        if crime_name in self.part_few:
                                            if f"{section}" in self.part_few[crime_name]:
                                                drop = 1
                                        if crime_name in self.part_few:
                                            if f"{section}" not in self.part_few[crime_name]:
                                                drop = 2
                                        G.add_edge(crime_node, section_node, edge_type=f"has_{crime_name}_{section}", crime_name = crime_name, part = drop)
                                        G.nodes[section_node]["texts"] = str(cont)
                    else:
                        
                        cont = content
                        if self.part_few is not None:
                            if crime_name in self.part_few:
                                print("1155**",section_node)
                                if f"{section}" in self.part_few[crime_name]:
                                    G.add_node(section_node, **default_node_attributes, node_type = "attribute", case_number = case['案件编号'], name = cont)
                                    drop = 0
                                    if crime_name in self.part_few:
                                        if f"{section}" in self.part_few[crime_name]:
                                            drop = 1
                                    if crime_name in self.part_few:
                                        if f"{section}" not in self.part_few[crime_name]:
                                            drop = 2
                                    G.add_edge(crime_node, section_node, edge_type = f"has_{crime_name}_{section}", crime_name = crime_name, part = drop)
                                    G.nodes[section_node]["texts"] = str(cont)
                        else:
                            print("1168**",section_node)
                            G.add_node(section_node, **default_node_attributes, node_type = "attribute", case_number = case['案件编号'], name = cont)
                            drop = 0
                            if crime_name in self.part_few:
                                if f"{section}" in self.part_few[crime_name]:
                                    drop = 1
                            if crime_name in self.part_few:
                                if f"{section}" not in self.part_few[crime_name]:
                                    drop = 2
                            G.add_edge(crime_node, section_node, edge_type = f"has_{crime_name}_{section}", crime_name = crime_name, part = drop)
                            G.nodes[section_node]["texts"] = str(cont)
                            
        for u, v, data in G.edges(data=True):
            edge_text = data.get("edge_type", "")
            data["edge_attr"] = edge_text
        return G

    def merge_graphs(self, graph_list):
        merged_graph = nx.DiGraph()
        offset = 0
        for graph in graph_list:
            mapping = {n: f"{n}_{offset}" for n in graph.nodes()}
            relabeled_graph = nx.relabel_nodes(graph, mapping)
            merged_graph = nx.compose(merged_graph, relabeled_graph)
            offset += 1

        return merged_graph
    
class Basic_Graph_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads = 1, model_type = "GAT"):
        super(Basic_Graph_Model, self).__init__()

        self.model_type = model_type
        if model_type == 'GAT':
            
            self.graph_model_l1 = GATConv(in_channels, hidden_channels, heads = 1, edge_dim = 768)
            self.graph_model_l2 = GATConv(in_channels, hidden_channels, heads = 1, edge_dim = 768)
        elif model_type == 'GATv2':
            self.graph_model_l1 = GATv2Conv(in_channels, hidden_channels, heads = 1, edge_dim = 768)
            self.graph_model_l2 = GATv2Conv(in_channels, hidden_channels, heads = 1, edge_dim = 768)
        elif model_type == 'CaseGNN':
            self.graph_model = CaseGNN(in_channels, hidden_channels, out_channels, dropout = 0.0, num_heads = 1)
        elif model_type == 'EAGATv2':
            self.graph_model_l1 = EAGATv2_EFWF(in_channels, hidden_channels, heads = 1, edge_dim = 768, wg_dim = 384)
            self.graph_model_l2 = EAGATv2_EFWF(in_channels, hidden_channels, heads = 1, edge_dim = 768, wg_dim = 64)

    def forward(self, data, drop_mode = False, drop_type = 'few'):
        x, edge_index, edge_attr, crime_name, drop, node_text = data.x, data.edge_index, data.edge_attr, data.crime_name, data.drop, data.name
        edge_type = data.edge_type
        drop = torch.tensor(drop)
        if drop_mode == True:
            if drop_type == 'few':
                mask = drop != torch.tensor(2)
            elif drop_type == 'most':
                mask = drop != torch.tensor(1)
            else:
                raise NotImplementedError
        else:
            mask = (drop == 0) | (drop == 1) | (drop == 2)
        crime_name = [relation for relation, mas in zip(crime_name, mask) if mas]
        mask = mask.to(device)
        filtered_edge_index = edge_index[: , mask]  # 只保留对应的 edge_index
        filtered_edge_attr = edge_attr[mask]  # 只保留对应的 edge_attr
        
        if self.model_type != 'CaseGNN':
            x1, (adj, alpha)  = self.graph_model_l1(x, filtered_edge_index, edge_attr = filtered_edge_attr, edge_type = edge_type, return_attention_weights=True)
            x2, (adj, alpha1) = self.graph_model_l2(x1, filtered_edge_index, edge_attr = filtered_edge_attr, edge_type = edge_type, return_attention_weights=True)
            
            x2 = F.relu(x1 + x2)
            return x2
        elif self.model_type == 'CaseGNN':
            x1 = self.graph_model(x, filtered_edge_index, filtered_edge_attr)
            return x1
        else:
            raise NotImplementedError

def margin_ranking_loss(real_scores, fake_scores, margin=0.1):
    real_expanded = real_scores.unsqueeze(1)  # [N, 1]
    fake_expanded = fake_scores.unsqueeze(0)  # [1, M]

    combined_real = real_expanded.expand(-1, fake_scores.size(0))  # [N, M]
    combined_fake = fake_expanded.expand(real_scores.size(0), -1)  # [N, M]

    labels = torch.ones(combined_real.size(0), combined_real.size(1), device=real_scores.device)  # [N, M]

    loss_fn = nn.MarginRankingLoss(margin=margin)
    loss = loss_fn(combined_real.flatten(), combined_fake.flatten(), labels.flatten())
    return loss

def ranking_loss_with_metrics(ranking_results, rels, k=3, margin=0.1):
    """
    结合MAP、NDCG和Precision@k来构造损失函数
    """
    
    # NDCG
    def NDCG_at_k(ranking_results, rels, k):
        Mean_NDCG = 0
        count = 0
        for query_id in ranking_results.keys():
            temp_NDCG = 0.
            temp_IDCG = 0.
            answer_list = sorted([2**(rels[query_id][candidate_id]-1) if (candidate_id in rels[query_id] and rels[query_id][candidate_id] >= 1) else 0. 
                                  for candidate_id in ranking_results[query_id]], reverse=True)
            for i, candidate_id in enumerate(ranking_results[query_id]):
                if i < k:
                    if candidate_id in rels[query_id]:
                        temp_gain = 2**(rels[query_id][candidate_id]-1) if rels[query_id][candidate_id] >= 1 else 0.
                        temp_NDCG += (temp_gain / math.log(i+2, 2))
                        temp_IDCG += (answer_list[i] / math.log(i+2, 2))
                else:
                    break
            if temp_IDCG > 0:
                Mean_NDCG += (temp_NDCG / temp_IDCG)
        Mean_NDCG /= (len(ranking_results.keys()) - count)
        return Mean_NDCG

    # MAP
    def MAP(ranking_results, rels):
        Mean_Average_Precision = 0
        count = 0
        for query_id in ranking_results.keys():
            if all([rels[query_id][did] != 3 for did in rels[query_id].keys()]):
                golden_labels = [2,3]
            else:
                golden_labels = [3]
            num_rel = 0
            Average_Precision = 0
            for i, candidate_id in enumerate(ranking_results[query_id]):
                if candidate_id in rels[query_id] and rels[query_id][candidate_id] in golden_labels:
                    num_rel += 1
                    Average_Precision += num_rel / (i + 1.0)
            if num_rel > 0:
                Average_Precision /= num_rel
            Mean_Average_Precision += Average_Precision
        Mean_Average_Precision /= (len(ranking_results.keys()) - count)
        return Mean_Average_Precision

    # Precision@k
    def Precision_at_k(ranking_results, rels, k):
        Precision_k = 0
        count = 0
        for query_id in ranking_results.keys():
            Precision = 0
            if query_id in rels:
                if all([rels[query_id][did] != 3 for did in rels[query_id].keys()]):
                    golden_labels = [2,3]
                else:
                    golden_labels = [3]
                for i, candidate_id in enumerate(ranking_results[query_id]):
                    if i + 1 <= k:
                        if candidate_id in rels[query_id] and rels[query_id][candidate_id] in golden_labels:
                            Precision += 1
                    else:
                        break
                Precision /= k
                Precision_k += Precision
            else:
                count += 1
        Precision_k /= (len(ranking_results.keys()) - count)
        return Precision_k

    # 计算各个指标
    NDCG = NDCG_at_k(ranking_results, rels, k)
    MAP_score = MAP(ranking_results, rels)
    Precision_score = Precision_at_k(ranking_results, rels, k)
    # print(NDCG,MAP_score,Precision_score,ranking_results)
    NDCG1 = NDCG_at_k(ranking_results, rels, 5)
    NDCG2 = NDCG_at_k(ranking_results, rels, 10)
    # 定义损失（较高的NDCG和MAP，较低的Precision都应该增加损失）
    loss = (1 - NDCG) + (1 - MAP_score) + (1 - Precision_score) + (1-NDCG1) + (1-NDCG2)
    return loss

if __name__ == '__main__':

    if args.dataset == 'lecardv2':
        label_json = "./labels/labelv2.json"
        dataset_dir = "./dataset/lecardv2"
    else:
        label_json = "./labels/labelv1.json"
        dataset_dir = "./dataset/lecard"
    
    with open(label_json,"r",encoding='utf-8')as f:
        label = json.load(f)
        f.close()

    files = os.listdir(dataset_dir)
    datas = []

    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(dataset_dir, file), "r", encoding = 'utf-8')as f:
                data = json.load(f)
                f.close()
            if len(data) > 10:
                datas.append(data)
    
    times = int(args.retrain_times)
    epochs = int(args.epochs)
    with open("./split_files/split_train_val.json", "r", encoding = 'utf-8')as f:
        train_test = json.load(f)
        f.close()
    exp_name = args.exp_name
    mode = args.mode
    if args.dataset == 'lecardv2':
        train_test_v2 = {"train": train_test["lecardv2"]["train"],"test": train_test["lecardv2"]["test"]} 
        part_few = args.attribute_file
        train_loader, val_loader = load_data(datas, train_test_v2, part_few = part_few)
        if mode != 'test':
            for n in range(times):
                print("times:",n)
                model = Basic_Graph_Model(in_channels=768, hidden_channels=768, out_channels=768, model_type = args.model)
                model.to(device)
                
                optimizer = optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
                if not os.path.exists("./experiment/"+exp_name):
                    os.mkdir("./experiment/"+exp_name) 
                if not os.path.exists("./experiment/"+exp_name+"/times"+str(n)):
                    os.mkdir("./experiment/"+exp_name+"/times"+str(n)) 
                loss_all_train = []
                loss_all_test = []
                for epoch in range(epochs):
                    print("epoch:",epoch)
                    model.train()
                    j1 = 0
                    k = 0
                    for batch in train_loader:
                        map1 = {}
                        sim_list = []
                        sim_list1 = []
                        total_loss = 0
                        temp = None
                        out_temp = None
                        optimizer.zero_grad()
                        
                        sim_3 = []
                        sim_other = []

                        data = batch[0]
                        out = model(data)
                        case_numbers = data.case_number 
                        case_query = case_numbers[0]
                        
                        out_temp = out[0]
                        case_numbers1 = list(filter(lambda x: x != case_query, case_numbers))
                        unique_cases = list(set(case_numbers1)) 
                        first_indices = []
                        for i in range(len(unique_cases)):
                            for j in range(len(case_numbers)):
                                if case_numbers[j] == unique_cases[i] and data.node_type[j] =="case":
                                    first_indices.append(j)
                                    break
                        first_node_features = torch.stack([out[i] for i in first_indices])
                        out1 = first_node_features
                        sims = F.cosine_similarity(out_temp.unsqueeze(0), out1, dim=1)

                        sim_list = []
                        sim_2 = []
                        sim_3 = []
                        sim_other = []
                        sim_other2 = []

                        unique_case_labels = torch.tensor([label[case_query].get(case, 0) for case in unique_cases], device=sims.device)

                        sim_list1 = sims.tolist()
                        for i, sim in enumerate(sims):
                            sim_entry = [sim, unique_cases[i], unique_case_labels[i].item()]
                            sim_list.append(sim_entry)

                        sim_2_mask = unique_case_labels >= 2
                        sim_3_mask = unique_case_labels == 3
                        sim_other_mask = ~sim_2_mask
                        sim_other2_mask = ~sim_3_mask

                        # 使用布尔索引获取相似度分组
                        if sim_2_mask.any():
                            sim_2 = sims[sim_2_mask]
                        if sim_3_mask.any():
                            sim_3 = sims[sim_3_mask]
                        if sim_other_mask.any():
                            sim_other = sims[sim_other_mask]
                        if sim_other2_mask.any():
                            sim_other2 = sims[sim_other2_mask]
                        j1 += 1
                        k += 1
                        loss = 0
                        result = {}
                        sim_list.sort(key = lambda x : -x[0])
                        result[case_query]={}
                        for item in sim_list:
                            result[case_query][item[1]] = item[0].item()
                            data_json = result
                        for key in data_json:
                            map1[key] = []
                            del_val = []
                            for vals in data_json[key]:
                                if key in label and vals not in label[key]:
                                    del_val.append(vals)
                                map1[key].append(int(vals))
                            for vals in del_val:
                                data_json[key].pop(vals)
                        
                        if len(sim_2)!=0 and len(sim_other)!=0:
                            print(data_json)
                            loss = margin_ranking_loss(sim_2,sim_other)+ranking_loss_with_metrics(data_json,label)
                            loss.backward()
                            print("loss_train:",j1,loss,case_query)
                            loss_all_train.append(loss)
                        sim_list.sort(key = lambda x:-x[0])
                        optimizer.step()
                        torch.cuda.empty_cache()
                    
                    print("train_loss_all",torch.mean(torch.stack(loss_all_train)))
                    print("val")
                    del out
                    model.eval()
                    loss_min = torch.tensor(10000).to(device)
                    j1 = 0
                    result = {}
                    with torch.no_grad():
                        for batch in val_loader:
                            result1 = {}
                            
                            map1 = {}
                            sim_list = []
                            sim_list1 = []
                            sim_3 = []
                            sim_other2 = []
                            total_loss = 0
                            temp = None
                            out_temp = None
                            data = batch[0]
                            out = model(data)
                            case_numbers = data.case_number 
                            case_query = case_numbers[0]
                            out_temp = out[0]
                            case_numbers1 = list(filter(lambda x: x != case_query, case_numbers))
                            unique_cases = list(set(case_numbers1)) 
                            first_indices =[]
                            for i in range(len(unique_cases)):
                                for j in range(len(case_numbers)):
                                    if case_numbers[j] == unique_cases[i] and data.node_type[j] =="case":
                                        first_indices.append(j)
                                        break
                            first_node_features = [out[i] for i in first_indices]
                            out1 = first_node_features
                            sim_list1 = []
                            sims = F.cosine_similarity(out_temp.unsqueeze(0), torch.stack(out1), dim=1)

                            # 初始化结果列表
                            sim_list = []
                            sim_2 = []
                            sim_3 = []
                            sim_other = []
                            sim_other2 = []

                            # 获取标签
                            unique_case_labels = torch.tensor([label[case_query].get(case, 0) for case in unique_cases], device=sims.device)

                            # 根据标签分类
                            sim_list1 = sims.tolist()  # 转为列表方便后续处理
                            for i, sim in enumerate(sims):
                                sim_entry = [sim, unique_cases[i], unique_case_labels[i].item()]
                                sim_list.append(sim_entry)

                            # 筛选符合条件的相似度
                            sim_2_mask = unique_case_labels >= 2
                            sim_3_mask = unique_case_labels == 3
                            sim_other_mask = ~sim_2_mask
                            sim_other2_mask = ~sim_3_mask

                            # 使用布尔索引获取相似度分组
                            if sim_2_mask.any():
                                sim_2 = sims[sim_2_mask]
                            if sim_3_mask.any():
                                sim_3 = sims[sim_3_mask]
                            if sim_other_mask.any():
                                sim_other = sims[sim_other_mask]
                            if sim_other2_mask.any():
                                sim_other2 = sims[sim_other2_mask]
                            sim_list.sort(key = lambda x:-x[0])
                            result1[case_query]={}
                            for item in sim_list:
                                result1[case_query][item[1]] = item[0].item()
                                data_json = result1
                            
                            for key in data_json:
                                map1[key] = []
                                del_val = []
                                for vals in data_json[key]:
                                    # print(key,vals)
                                    if key in label and vals not in label[key]:
                                        del_val.append(vals)
                                    map1[key].append(int(vals))
                                for vals in del_val:
                                    data_json[key].pop(vals)
                            if len(sim_2)!=0 and len(sim_other)!=0:
                                loss1 = margin_ranking_loss(sim_2, sim_other) + ranking_loss_with_metrics(data_json, label)
                                loss_all_test.append(loss1)
                                print("loss_test:",j1,loss1,case_query)
                            j1+=1
                            k+=1
                            sim_list.sort(key = lambda x:-x[0])
                            result[case_query]={}
                            for item in sim_list:
                                result[case_query][item[1]] = item[0].item()
                            torch.cuda.empty_cache()
                            with open("./experiment/"+exp_name+"/times"+str(n)+"/"+str(epoch)+"result.json","w",encoding='utf-8')as f:
                                json.dump(result, f, ensure_ascii = False, indent = 4)
                        print("test_loss_all", torch.mean(torch.stack(loss_all_test)))   
                        if torch.mean(torch.stack(loss_all_test)) < loss_min:
                            loss_min = torch.mean(torch.stack(loss_all_test))
                            torch.save(model, "./checkpoints/"+exp_name+".pth") 
                
        else:
            result_all = {}
            model = torch.load("/home/zhaokx/DB-GAF/checkpoints/eagatv2_drop.pth")
            model.to(device)
            k = 0
            with open("/home/zhaokx/DB-GAF/split_files/v2_2_u.json","r",encoding='utf-8')as f:
                model.unstable = json.load(f)
                f.close()
            with torch.no_grad():
                for batch in val_loader:
                    sim_list = []
                    model.eval()
                    data = batch[0]
                    data = data.to(device)
                    out = model(data)
                    out_temp = out[0].unsqueeze(0)
                    case_numbers = data.case_number 
                    case_query = case_numbers[0]
                    unique_cases = []
                    j = 0
                    for cn in case_numbers:
                        if data.node_type[j] =="case" and cn !=case_query:
                            unique_cases.append(cn)
                        j+=1
                    first_indices =[]
                    for i in range(len(unique_cases)):
                        for j in range(len(case_numbers)):
                            if case_numbers[j] == unique_cases[i] and data.node_type[j] =="case":
                                first_indices.append(j)
                                break
                    first_node_features = torch.stack([out[i] for i in first_indices])
                    out1 = first_node_features

                    sims = F.cosine_similarity(out_temp, out1, dim=1)
                    unique_case_labels = torch.tensor([label[case_query].get(case, 0) for case in unique_cases], device=sims.device)
                    for i, sim in enumerate(sims):
                        sim_entry = [sims[i], unique_cases[i], unique_case_labels[i].item()]
                        sim_list.append(sim_entry)
                    sim_list.sort(key = lambda x:-x[0])
                    result_all[case_query]={}
                    for item in sim_list:
                        result_all[case_query][item[1]] = item[0].item()
                    k+=1
                    print(k)
                    # break
                with open("result"+".json","w",encoding='utf-8')as f:
                    json.dump(result_all,f,ensure_ascii=False, indent=4)
            checkpoint = torch.load("/home/zhaokx/DB-GAF/checkpoints/v2_in_2_l_pe_train_val.pth")
            rename_mapping = {
                "gat1.att":"graph_model_l1.att",
                "gat1.bias":"graph_model_l1.bias",
                "gat1.lin_l.weight": "graph_model_l1.lin_l.weight",
                "gat1.lin_l.bias": "graph_model_l1.lin_l.bias",
                "gat1.lin_r.weight": "graph_model_l1.lin_r.weight",
                "gat1.lin_r.bias": "graph_model_l1.lin_r.bias",
                "gat1.lin_edge.weight": "graph_model_l1.lin_edge.weight",
                "gat1.edge_encode.weight":"graph_model_l1.edge_encode.weight",
                "gat1.edge_encode.bias":"graph_model_l1.edge_encode.bias",
                "gat1.weight.weight":"graph_model_l1.weight.weight",
                "gat1.weight.bias":"graph_model_l1.weight.bias",
                "gat1.output_proj.weight":"graph_model_l1.output_proj.weight",
                "gat1.output_proj.bias":"graph_model_l1.output_proj.bias",
                "gat1.weight_generator.0.weight":"graph_model_l1.weight_generator.0.weight",
                "gat1.weight_generator.0.bias":"graph_model_l1.weight_generator.0.bias",
                "gat1.weight_generator.2.weight":"graph_model_l1.weight_generator.2.weight",
                "gat1.weight_generator.2.bias":"graph_model_l1.weight_generator.2.bias",
                
                "gat4.att":"graph_model_l2.att",
                "gat4.bias":"graph_model_l2.bias",
                "gat4.lin_l.weight": "graph_model_l2.lin_l.weight",
                "gat4.lin_l.bias": "graph_model_l2.lin_l.bias",
                "gat4.lin_r.weight": "graph_model_l2.lin_r.weight",
                "gat4.lin_r.bias": "graph_model_l2.lin_r.bias",
                "gat4.lin_edge.weight": "graph_model_l2.lin_edge.weight",
                "gat4.edge_encode.weight":"graph_model_l2.edge_encode.weight",
                "gat4.edge_encode.bias":"graph_model_l2.edge_encode.bias",
                "gat4.weight.weight":"graph_model_l2.weight.weight",
                "gat4.weight.bias":"graph_model_l2.weight.bias",
                "gat4.output_proj.weight":"graph_model_l2.output_proj.weight",
                "gat4.output_proj.bias":"graph_model_l2.output_proj.bias",
                "gat4.weight_generator.0.weight":"graph_model_l2.weight_generator.0.weight",
                "gat4.weight_generator.0.bias":"graph_model_l2.weight_generator.0.bias",
                "gat4.weight_generator.2.weight":"graph_model_l2.weight_generator.2.weight",
                "gat4.weight_generator.2.bias":"graph_model_l2.weight_generator.2.bias",
               
            }
            fields_to_remove = []
            new_state_dict = {}
            model = Basic_Graph_Model(in_channels=768, hidden_channels=768, out_channels=768, model_type = args.model)
            for old_key, value in checkpoint.named_parameters():
                print(old_key)
                if old_key in fields_to_remove:
                    continue  # 直接跳过不需要的字段
                new_key = rename_mapping.get(old_key, old_key)  # 如果有映射关系就重命名，否则保持原键名
                new_state_dict[new_key] = value
                if new_key in model.state_dict():
                    print(new_key,'1677')
                    model.state_dict()[new_key].copy_(value.data)
            torch.save(model,"./checkpoints/eagatv2_drop.pth") 
    elif args.dataset == 'lecard':
        if args.fold == 0:  
            train_test_v1 = {"train": train_test["lecard"]["fold0_train"], "test": train_test["lecard"]["fold0_test"]} 
            train_loader, val_loader = load_data(datas, train_test_v1)
            for n in range(times):
                print("times:",n)
                model = Basic_Graph_Model(in_channels=768, hidden_channels=768, out_channels=768, model_type = args.model)
                model.to(device)
                optimizer = optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
                if not os.path.exists("./experiment/" + exp_name):
                    os.mkdir("./experiment/" + exp_name) 
                if not os.path.exists("./experiment/" + exp_name + "/times" + str(n)):
                    os.mkdir("./experiment/" + exp_name + "/times" + str(n)) 
                if not os.path.exists("./experiment/" + exp_name + "/times" + str(n) + "/fold" + str(0)):
                    os.mkdir("./experiment/" + exp_name + "/times" + str(n) + "/fold" + str(0)) 
                loss_all_train = []
                loss_all_test = []
                for epoch in range(epochs):
                    print("epoch:",epoch)
                    model.train()
                    j1 = 0
                    k = 0
                    result = {}
                    for batch in train_loader:
                        sim_list = []
                        sim_list1 = []
                        total_loss = 0
                        temp = None
                        out_temp = None
                        optimizer.zero_grad()
                        
                        sim_3 = []
                        sim_other = []
                        data = batch[0]
                        out = model(data)
                        case_numbers = data.case_number 
                        case_query = case_numbers[0]
                        
                        out_temp = out[0]
                        case_numbers1 = list(filter(lambda x: x != case_query, case_numbers))
                        unique_cases = list(set(case_numbers1)) 
                        first_indices = []
                        for i in range(len(unique_cases)):
                            for j in range(len(case_numbers)):
                                if case_numbers[j] == unique_cases[i] and data.node_type[j] =="case":
                                    first_indices.append(j)
                                    break
                        first_node_features = torch.stack([out[i] for i in first_indices])
                        out1 = first_node_features
                        sims = F.cosine_similarity(out_temp.unsqueeze(0), out1, dim=1)

                        sim_list = []
                        sim_2 = []
                        sim_3 = []
                        sim_other = []
                        sim_other2 = []

                        unique_case_labels = torch.tensor([label[case_query].get(case, 0) for case in unique_cases], device=sims.device)

                        sim_list1 = sims.tolist()  # 转为列表方便后续处理
                        for i, sim in enumerate(sims):
                            sim_entry = [sim, unique_cases[i], unique_case_labels[i].item()]
                            sim_list.append(sim_entry)

                        sim_2_mask = unique_case_labels >= 2
                        sim_3_mask = unique_case_labels == 3
                        sim_other_mask = ~sim_2_mask
                        sim_other2_mask = ~sim_3_mask

                        # 使用布尔索引获取相似度分组
                        if sim_2_mask.any():
                            sim_2 = sims[sim_2_mask]
                        if sim_3_mask.any():
                            sim_3 = sims[sim_3_mask]
                        if sim_other_mask.any():
                            sim_other = sims[sim_other_mask]
                        if sim_other2_mask.any():
                            sim_other2 = sims[sim_other2_mask]
                        j1 += 1
                        k += 1
                        loss1 = 0
                        sim_list.sort(key = lambda x : -x[0])
                        result[case_query]={}
                        for item in sim_list:
                            result[case_query][item[1]] = item[0].item()
                            data_json = result
                        for key in data_json:
                            map1[key] = []
                            del_val = []
                            for vals in data_json[key]:
                                if key in label_dic and vals not in label_dic[key]:
                                    del_val.append(vals)
                                map1[key].append(int(vals))
                            for vals in del_val:
                                data_json[key].pop(vals)
                        
                        if len(sim_3)!=0 and len(sim_other2)!=0:
                            loss = margin_ranking_loss(sim_3,sim_other2)+ranking_loss_with_metrics(data_json,label_dic)
                            loss.backward()
                            print("loss_train:",j1,loss1,case_query)
                            loss_all_train.append(loss1)
                        sim_list.sort(key = lambda x:-x[0])
                        optimizer.step()
                        torch.cuda.empty_cache()
                    
                    print("train_loss_all",torch.mean(torch.stack(loss_all_train)))
                    print("val")
                    del out
                    model.eval()
                    loss_min = torch.tensor(10000).to(device)
                    j1 = 0
                    result = {}
                    result1 = {}
                    with torch.no_grad():
                        for batch in val_loader:
                            sim_list = []
                            sim_list1 = []
                            sim_3 = []
                            sim_other2 = []
                            total_loss = 0
                            temp = None
                            out_temp = None
                            data = batch[0]
                            out = model(data)
                            case_numbers = data.case_number 
                            case_query = case_numbers[0]
                            out_temp = out[0]
                            case_numbers1 = list(filter(lambda x: x != case_query, case_numbers))
                            unique_cases = list(set(case_numbers1)) 
                            first_indices =[]
                            for i in range(len(unique_cases)):
                                for j in range(len(case_numbers)):
                                    if case_numbers[j] == unique_cases[i] and data.node_type[j] =="case":
                                        first_indices.append(j)
                                        break
                            first_node_features = [out[i] for i in first_indices]
                            out1 = first_node_features
                            sim_list1 = []
                            sims = F.cosine_similarity(out_temp.unsqueeze(0), torch.stack(out1), dim=1)

                            # 初始化结果列表
                            sim_list = []
                            sim_2 = []
                            sim_3 = []
                            sim_other = []
                            sim_other2 = []

                            # 获取标签
                            unique_case_labels = torch.tensor([label[case_query].get(case, 0) for case in unique_cases], device=sims.device)

                            # 根据标签分类
                            sim_list1 = sims.tolist()  # 转为列表方便后续处理
                            for i, sim in enumerate(sims):
                                sim_entry = [sim, unique_cases[i], unique_case_labels[i].item()]
                                sim_list.append(sim_entry)

                            # 筛选符合条件的相似度
                            sim_2_mask = unique_case_labels >= 2
                            sim_3_mask = unique_case_labels == 3
                            sim_other_mask = ~sim_2_mask
                            sim_other2_mask = ~sim_3_mask

                            # 使用布尔索引获取相似度分组
                            if sim_2_mask.any():
                                sim_2 = sims[sim_2_mask]
                            if sim_3_mask.any():
                                sim_3 = sims[sim_3_mask]
                            if sim_other_mask.any():
                                sim_other = sims[sim_other_mask]
                            if sim_other2_mask.any():
                                sim_other2 = sims[sim_other2_mask]
                            result1[case_query]={}
                            for item in sim_list:
                                result1[case_query][item[1]] = item[0].item()
                                data_json = result1
                            for key in data_json:
                                map1[key] = []
                                del_val = []
                                for vals in data_json[key]:
                                    # print(key,vals)
                                    if key in label_dic and vals not in label_dic[key]:
                                        del_val.append(vals)
                                    map1[key].append(int(vals))
                                for vals in del_val:
                                    data_json[key].pop(vals)
                            if len(sim_3)!=0 and len(sim_other2)!=0:
                                loss1 = margin_ranking_loss(sim_3,sim_other2)+ranking_loss_with_metrics(data_json,label_dic)
                                loss_all_test.append(loss1)
                                print("loss_test:",j1,loss1,case_query)
                            j1+=1
                            k+=1
                            sim_list.sort(key = lambda x: -x[0])
                            result[case_query]={}
                            for item in sim_list:
                                result[case_query][item[1]] = item[0].item()
                            torch.cuda.empty_cache()
                            with open("./experiment/" + exp_name + "/times" + str(n) + "/fold" + str(0) + "/" + str(epoch) + "result.json", "w", encoding = 'utf-8') as f:
                                json.dump(result, f, ensure_ascii = False, indent = 4)
                        print("test_loss_all",torch.mean(torch.stack(loss_all_test)))   
                        if torch.mean(torch.stack(loss_all_test)) < loss_min:
                            loss_min = torch.mean(torch.stack(loss_all_test))
                            torch.save(model,"./checkpoints/" + exp_name + "_fold0.pth") 
                
        elif args.fold == 1:
            train_test_v1 = {"train": train_test["lecard"]["fold1_train"],"test": train_test["lecard"]["fold1_test"]} 
            train_loader, val_loader = load_data(datas, train_test_v1)

            for n in range(times):
                print("times:",n)
                model = Basic_Graph_Model(in_channels=768, hidden_channels=768, out_channels=768, model_type = args.model)
                model.to(device)
                optimizer = optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
                if not os.path.exists("./experiment/"+exp_name):
                    os.mkdir("./experiment/"+exp_name) 
                if not os.path.exists("./experiment/"+exp_name+"/times"+str(n)):
                    os.mkdir("./experiment/"+exp_name+"/times"+str(n)) 
                if not os.path.exists("./experiment/"+exp_name+"/times"+str(n)+"/fold"+str(1)):
                    os.mkdir("./experiment/"+exp_name+"/times"+str(n)+"/fold"+str(1)) 
                
                loss_all_train = []
                loss_all_test = []
                for epoch in range(epochs):
                    print("epoch:",epoch)
                    model.train()
                    j1 = 0
                    k = 0
                    result = {}
                    for batch in train_loader:
                        sim_list = []
                        sim_list1 = []
                        total_loss = 0
                        temp = None
                        out_temp = None
                        optimizer.zero_grad()
                        
                        sim_3 = []
                        sim_other = []
                        data = batch
                        out = model(data)
                        case_numbers = data.case_number 
                        case_query = case_numbers[0]
                        
                        out_temp = out[0]
                        case_numbers1 = list(filter(lambda x: x != case_query, case_numbers))
                        unique_cases = list(set(case_numbers1)) 
                        first_indices = []
                        for i in range(len(unique_cases)):
                            for j in range(len(case_numbers)):
                                if case_numbers[j] == unique_cases[i] and data.node_type[j] =="case":
                                    first_indices.append(j)
                                    break
                        first_node_features = torch.stack([out[i] for i in first_indices])
                        out1 = first_node_features
                        sims = F.cosine_similarity(out_temp.unsqueeze(0), out1, dim=1)

                        sim_list = []
                        sim_2 = []
                        sim_3 = []
                        sim_other = []
                        sim_other2 = []

                        unique_case_labels = torch.tensor([label[case_query].get(case, 0) for case in unique_cases], device=sims.device)

                        sim_list1 = sims.tolist()  # 转为列表方便后续处理
                        for i, sim in enumerate(sims):
                            sim_entry = [sim, unique_cases[i], unique_case_labels[i].item()]
                            sim_list.append(sim_entry)

                        sim_2_mask = unique_case_labels >= 2
                        sim_3_mask = unique_case_labels == 3
                        sim_other_mask = ~sim_2_mask
                        sim_other2_mask = ~sim_3_mask

                        # 使用布尔索引获取相似度分组
                        if sim_2_mask.any():
                            sim_2 = sims[sim_2_mask]
                        if sim_3_mask.any():
                            sim_3 = sims[sim_3_mask]
                        if sim_other_mask.any():
                            sim_other = sims[sim_other_mask]
                        if sim_other2_mask.any():
                            sim_other2 = sims[sim_other2_mask]
                        j1 += 1
                        k += 1
                        loss1 = 0
                        sim_list.sort(key = lambda x : -x[0])
                        result[case_query]={}
                        for item in sim_list:
                            result[case_query][item[1]] = item[0].item()
                            data_json = result
                        for key in data_json:
                            map1[key] = []
                            del_val = []
                            for vals in data_json[key]:
                                if key in label_dic and vals not in label_dic[key]:
                                    del_val.append(vals)
                                map1[key].append(int(vals))
                            for vals in del_val:
                                data_json[key].pop(vals)
                        
                        if len(sim_3)!=0 and len(sim_other2)!=0:
                            loss = margin_ranking_loss(sim_3,sim_other2)+ranking_loss_with_metrics(data_json,label_dic)
                            loss.backward()
                            print("loss_train:",j1,loss1,case_query)
                            loss_all_train.append(loss1)
                        sim_list.sort(key = lambda x:-x[0])
                        optimizer.step()
                        torch.cuda.empty_cache()
                    
                    print("train_loss_all",torch.mean(torch.stack(loss_all_train)))
                    print("val")
                    del out
                    model.eval()
                    loss_min = torch.tensor(10000).to(device)
                    j1 = 0
                    result = {}
                    result1 = {}
                    with torch.no_grad():
                        for batch in val_loader:
                            sim_list = []
                            sim_list1 = []
                            sim_3 = []
                            sim_other2 = []
                            total_loss = 0
                            temp = None
                            out_temp = None
                            data = batch
                            out = model(data)
                            case_numbers = data.case_number 
                            case_query = case_numbers[0]
                            out_temp = out[0]
                            case_numbers1 = list(filter(lambda x: x != case_query, case_numbers))
                            unique_cases = list(set(case_numbers1)) 
                            first_indices =[]
                            for i in range(len(unique_cases)):
                                for j in range(len(case_numbers)):
                                    if case_numbers[j] == unique_cases[i] and data.node_type[j] =="case":
                                        first_indices.append(j)
                                        break
                            first_node_features = [out[i] for i in first_indices]
                            out1 = first_node_features
                            sim_list1 = []
                            sims = F.cosine_similarity(out_temp.unsqueeze(0), torch.stack(out1), dim=1)

                            # 初始化结果列表
                            sim_list = []
                            sim_2 = []
                            sim_3 = []
                            sim_other = []
                            sim_other2 = []

                            # 获取标签
                            unique_case_labels = torch.tensor([label[case_query].get(case, 0) for case in unique_cases], device=sims.device)

                            # 根据标签分类
                            sim_list1 = sims.tolist()  # 转为列表方便后续处理
                            for i, sim in enumerate(sims):
                                sim_entry = [sim, unique_cases[i], unique_case_labels[i].item()]
                                sim_list.append(sim_entry)

                            # 筛选符合条件的相似度
                            sim_2_mask = unique_case_labels >= 2
                            sim_3_mask = unique_case_labels == 3
                            sim_other_mask = ~sim_2_mask
                            sim_other2_mask = ~sim_3_mask

                            # 使用布尔索引获取相似度分组
                            if sim_2_mask.any():
                                sim_2 = sims[sim_2_mask]
                            if sim_3_mask.any():
                                sim_3 = sims[sim_3_mask]
                            if sim_other_mask.any():
                                sim_other = sims[sim_other_mask]
                            if sim_other2_mask.any():
                                sim_other2 = sims[sim_other2_mask]
                            result1[case_query]={}
                            for item in sim_list:
                                result1[case_query][item[1]] = item[0].item()
                                data_json = result1
                            for key in data_json:
                                map1[key] = []
                                del_val = []
                                for vals in data_json[key]:
                                    # print(key,vals)
                                    if key in label_dic and vals not in label_dic[key]:
                                        del_val.append(vals)
                                    map1[key].append(int(vals))
                                for vals in del_val:
                                    data_json[key].pop(vals)
                            if len(sim_3)!=0 and len(sim_other2)!=0:
                                loss1 = margin_ranking_loss(sim_3,sim_other2)+ranking_loss_with_metrics(data_json,label_dic)
                                loss_all_test.append(loss1)
                                print("loss_test:",j1,loss1,case_query)
                            j1+=1
                            k+=1
                            sim_list.sort(key = lambda x:-x[0])
                            result[case_query]={}
                            for item in sim_list:
                                result[case_query][item[1]] = item[0].item()
                            torch.cuda.empty_cache()
                            with open("./experiment/"+exp_name+"/times"+str(n)+"/fold"+str(1)+"/"+str(epoch)+"result.json","w",encoding='utf-8')as f:
                                json.dump(result, f, ensure_ascii = False, indent = 4)
                        print("test_loss_all",torch.mean(torch.stack(loss_all_test)))   
                        if torch.mean(torch.stack(loss_all_test))<loss_min:
                            loss_min = torch.mean(torch.stack(loss_all_test))
                            torch.save(model,"./checkpoints/"+exp_name+"_fold1.pth") 
            







        

















