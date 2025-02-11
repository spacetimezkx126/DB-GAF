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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description = 'DB-GAF for lcr')
parser.add_argument('--dataset', type = str, default = 'lecardv2', 
                    help = 'dataset_name')  
parser.add_argument('--fold', type = str, default = 0,
                    help = 'indicate fold for lecard (0,1)')
parser.add_argument('--exp_name', type = str, default = 'exp_1', 
                    help = 'experiment_name')  
parser.add_argument('--retrain_times', type = str, default = 10, 
                    help = 'training times')
parser.add_argument('--epochs', type = str, default = 10, 
                    help = 'epochs')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("./models--CSHaitao--SAILER_zh")
sailer_model = AutoModel.from_pretrained("./models--CSHaitao--SAILER_zh").to(device)

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
    kfold_loaders = [] 

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

def ranking_loss_with_metrics(ranking_results, rels, k=3, margin=0.1):
    
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

    NDCG = NDCG_at_k(ranking_results, rels, k)
    MAP_score = MAP(ranking_results, rels)
    Precision_score = Precision_at_k(ranking_results, rels, k)
    NDCG1 = NDCG_at_k(ranking_results, rels, 5)
    NDCG2 = NDCG_at_k(ranking_results, rels, 10)
    loss = (1 - NDCG) + (1 - MAP_score) + (1 - Precision_score) + (1-NDCG1) + (1-NDCG2)
    return loss

def margin_ranking_loss(real_scores, fake_scores, margin=0.1):
    real_expanded = real_scores.unsqueeze(1)  # [N, 1]
    fake_expanded = fake_scores.unsqueeze(0)  # [1, M]

    combined_real = real_expanded.expand(-1, fake_scores.size(0))  # [N, M]
    combined_fake = fake_expanded.expand(real_scores.size(0), -1)  # [N, M]

    labels = torch.ones(combined_real.size(0), combined_real.size(1), device=real_scores.device)  # [N, M]

    loss_fn = nn.MarginRankingLoss(margin=margin)

    loss = loss_fn(combined_real.flatten(), combined_fake.flatten(), labels.flatten())

    return loss

class EAGATv2_EWA(MessagePassing):
    r"""The EAGATv2-EWA operator is an extension of the GATv2 operator from the  
    "How Attentive are Graph Attention Networks?" <https://arxiv.org/abs/2105.14491>_ paper,  
    which addresses the static attention issue in the standard :class:`~torch_geometric.conv.GATConv` layer.  
    EAGATv2-EWA enhances the GATv2 framework by introducing an Edge Weighting Attention (EWA) mechanism,  
    which dynamically adjusts edge weights to better leverage edge feature information, thereby improving  
    the performance of graph neural networks.  

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

        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr, edge_weight=edge_weight)
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
                    edge_weight: OptTensor = None) -> Tensor:
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

        if edge_weight is not None:
            alpha = alpha * edge_weight.view(-1, 1)

        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
        

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class CaseDataset(Dataset):
    def __init__(self, data, dataset = 'lecard', part_few = 'None'):
        self.data_list = data
        self.device = torch.device("cuda:0")
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
        
        with open("./dataset/lecardv2_contrast_graph.pkl",'rb')as f:
            # self.cached_data = pkl.load(f)
        self.cached_data = []
        self.cache_all_data()
    def __len__(self):
        return len(self.data_list)

    def cache_all_data(self):
        for idx, case_batch in enumerate(self.data_list):
            graph_list = [self.build_graph(case) for case in case_batch]
            merged_graph = self.merge_graphs(graph_list)
            data = from_networkx(merged_graph)
            data.x = torch.zeros(len(merged_graph.nodes),768)
            case_mask = torch.tensor([bool(n!='crime1') for n in data.node_type])  
            data.x[case_mask] = encode_text([merged_graph.nodes[n]["texts"] for n in merged_graph.nodes if merged_graph.nodes[n]["node_type"] != "crime1"]).detach().cpu()
            case_mask = torch.tensor([bool(n=='crime1') for n in data.node_type]) 
            data.x[case_mask] = encode_text([merged_graph.nodes[n]["texts"] for n in merged_graph.nodes if merged_graph.nodes[n]["node_type"] == "crime1"],length=300).detach().cpu()
            data.edge_attr = encode_text([merged_graph[u][v]["edge_attr"] for u, v in merged_graph.edges]).detach().cpu()
            data.crime_name = [merged_graph[u][v]["crime_name"] for u, v in merged_graph.edges]
            self.cached_data.append(data)
        save_to_pkl(self.cached_data, 'lecardv2_contrast_graph.pkl')
    def __getitem__(self, idx):
        return self.cached_data[idx].to(self.device)
    
    def build_graph(self, case):
        G = nx.DiGraph()
        default_node_attributes = {}
        pad_size = 64
        PAD = '<PAD>'
        UNK = '<UNK>'
        case_id = f"case_{case['案件编号']}"
        origin_text = None
        if case['案件编号'] in self.data_doc:
            origin_text = self.data_doc[case['案件编号']]
        elif case['案件编号'] in self.data_query:
            origin_text = self.data_query[case['案件编号']]
        
        G.add_node(case_id, **default_node_attributes, node_type="case",case_number=case['案件编号'], name=case_id)
        G.nodes[case_id]["texts"] = "None"
        branch1_node = f"branch1_attributes_{case['案件编号']}"
        branch2_node = f"branch2_attributes_{case['案件编号']}"
        G.add_node(branch1_node, **default_node_attributes, node_type="virtual", case_number=case['案件编号'], name="important")
        G.nodes[branch1_node]["texts"] = "important"
        G.add_edge(case_id, branch1_node, edge_type="has_virtual",crime_name="None")
        G.add_node(branch2_node, **default_node_attributes, node_type="virtual", case_number=case['案件编号'], name="unimportant")
        G.nodes[branch2_node]["texts"] = "unimportant"
        G.add_edge(case_id, branch2_node, edge_type="has_virtual",crime_name="None")
        zuiming = []
        
        for crime_info in case["不同罪名"]:
            crime_name = crime_info["罪名"]
            if crime_name not in zuiming:
                zuiming.append(crime_name)

                if crime_name in origin_text:
                    crime_node1 = crime_name
                    if "经过：" in origin_text[crime_name]:
                        G.add_node(crime_node1, **default_node_attributes, node_type="crime1",case_number=case['案件编号'],name=crime_name+":"+origin_text[crime_name])
                        G.add_edge(case_id, crime_node1, edge_type="has_crime1",crime_name=crime_name)
                        G.nodes[crime_node1]["texts"] = crime_name+":"+origin_text[crime_name]
                
                crime_node_branch1 = f"crime_important_{crime_name}_{case['案件编号']}"
                G.add_node(crime_node_branch1, **default_node_attributes, node_type="crime",case_number=case['案件编号'],name=crime_name)
                G.add_edge(branch1_node, crime_node_branch1, edge_type="has_crime"+crime_name,crime_name=crime_name)
                G.nodes[crime_node_branch1]["texts"] = crime_name

                crime_node_branch2 = f"crime_unimportant_{crime_name}_{case['案件编号']}"
                G.add_node(crime_node_branch2, **default_node_attributes, node_type="crime",case_number=case['案件编号'],name=crime_name)
                G.add_edge(branch2_node, crime_node_branch2, edge_type="has_crime"+crime_name,crime_name=crime_name)
                G.nodes[crime_node_branch2]["texts"] = crime_name
                for section, content in crime_info.items():
                    section_node = f"{section}_{crime_name}_{case['案件编号']}"
                    if isinstance(content,list):
                        for cont in content:
                            G.add_node(section_node, **default_node_attributes, node_type="attribute",case_number=case['案件编号'], name=cont)
                            G.add_edge(crime_node_branch1, section_node, edge_type=f"has_{crime_name}_important_{section}",crime_name=crime_name)
                            G.add_edge(crime_node_branch2, section_node, edge_type=f"has_{crime_name}_unimportant_{section}",crime_name=crime_name)
                            G.nodes[section_node]["texts"] = str(cont)
                    else:
                        cont = content
                        G.add_node(section_node, **default_node_attributes, node_type="attribute",case_number=case['案件编号'], name=cont)
                        G.add_edge(crime_node_branch1, section_node, edge_type=f"has_{crime_name}_important_{section}",crime_name=crime_name)
                        G.add_edge(crime_node_branch2, section_node, edge_type=f"has_{crime_name}_unimportant_{section}",crime_name=crime_name)
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
    
class EdgeWeighted(nn.Module):
    def __init__(self, in_channels, heads=1):
        super(EdgeWeighted, self).__init__()
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, edge_attr):
        edge_weight = self.compute_edge_weight(edge_attr)
        return edge_weight
    
    def compute_edge_weight(self, edge_attr):
        edge_weight = self.edge_mlp(edge_attr).squeeze(-1)
        return edge_weight

class DB_GAF(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1):
        super(DB_GAF, self).__init__()

        self.graph_model_l1 = EAGATv2_EWA(in_channels, hidden_channels, heads = 1, edge_dim = 768)
        self.graph_model_l2 = EAGATv2_EWA(in_channels, hidden_channels, heads = 1, edge_dim = 768)
        self.graph_model_l3 = EAGATv2_EWA(in_channels, hidden_channels, heads = 1, edge_dim = 768)

        self.edge_weight = EdgeWeighted(768)
        self.edge_weight2 = EdgeWeighted(768)
        self.edge_weight3 = EdgeWeighted(768)

    def forward(self, data):
    
        attention = []
        attention2 = []
        attention3 = []

        x, edge_index, edge_attr, crime_name, edge_type = data.x, data.edge_index, data.edge_attr, data.crime_name, data.edge_type

        edge_weight = self.edge_weight(edge_attr)
        x1, (adj, alpha) = self.graph_model_l1(x, edge_index, edge_attr, edge_weight = edge_weight, return_attention_weights=True)

        i = 0
        for etype in edge_type:
            attention.append([crime_name[i],etype,edge_weight[i].detach().cpu(),alpha[i][0].detach().cpu().item()])
            i += 1
        
        edge_weight2 = self.edge_weight2(edge_attr)

        x2,(adj, alpha)= self.graph_model_l2(x1, edge_index,edge_attr,edge_weight = edge_weight2, return_attention_weights=True)
        x2 = F.relu(x2+x1)

        i = 0
        for etype in edge_type:
            attention2.append([crime_name[i],etype,edge_weight2[i].detach().cpu(),alpha[i][0].detach().cpu().item()])
            i += 1
        
        edge_weight3 = self.edge_weight3(edge_attr)
        x3, (adj, alpha) = self.graph_model_l3(x2,edge_index,edge_attr,edge_weight = edge_weight3,return_attention_weights=True)
        x3 = F.relu(x3+x2)

        i = 0
        for etype in edge_type:
            attention3.append([crime_name[i],etype,edge_weight3[i].detach().cpu(),alpha[i][0].detach().cpu().item()])
            i += 1
        
        return x3,attention,attention2,attention3

def load_data(datas,train_test):
    train_ratio = 0.8
    case_dataset = CaseDataset(datas, dataset = 'lecardv2')
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
    
    if args.dataset == 'lecardv2':
        label_dic = json.load(open("./labels/labelv2.json","r"))
        train_test_v2 = {"train": train_test["lecardv2"]["train"],"test": train_test["lecardv2"]["test"]} 
        train_loader, val_loader = load_data(datas, train_test_v2)
        loss_min = torch.tensor(10000).to(device)
        for n in range(times):
            result_all = {}
            print("times:",n)
            model = DB_GAF(in_channels=768, hidden_channels=768, out_channels=768)
            model.to(device)
            optimizer = optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
            if not os.path.exists("./experiment/"+exp_name):
                os.mkdir("./experiment/"+exp_name)
            if not os.path.exists("./experiment/"+exp_name+"/times"+str(n)):
                os.mkdir("./experiment/"+exp_name+"/times"+str(n))
            loss_all_train = []
            loss_all_test = []
            attention_train = {}
            attention_val = {}
            attention_train2 = {}
            attention_val2 = {}
            attention_train3 = {}
            attention_val3 = {}
            for epoch in range(epochs):
                print("epoch:",epoch)
                model.train()
                for batch in train_loader:
                    map1 = {}
                    result = {}
                    sim_list = []
                    sim_list1 = []
                    total_loss = 0
                    temp = None
                    out_temp = None
                    optimizer.zero_grad()

                    sim_3 = []
                    sim_other = []

                    data = batch[0]
                    output = model(data)
                    out, attention, attention2, attention3 = output[0], output[1], output[2], output[3]
                    for attn in attention:
                        if (attn[0],attn[1]) not in attention_train:
                            attention_train[(attn[0],attn[1])] = []
                            # attention_train_gat[(attn[0],attn[1])] = []
                        attention_train[(attn[0],attn[1])].append(attn[2])
                        # attention_train_gat[(attn[0],attn[1])].append(attn[3])
                    for attn in attention2:
                        if (attn[0],attn[1]) not in attention_train2:
                            attention_train2[(attn[0],attn[1])] = []
                            # attention_train2_gat[(attn[0],attn[1])] = []
                        attention_train2[(attn[0],attn[1])].append(attn[2])
                        # attention_train2_gat[(attn[0],attn[1])].append(attn[3])
                    for attn in attention3:
                        if (attn[0],attn[1]) not in attention_train3:
                            attention_train3[(attn[0],attn[1])] = []
                            # attention_train3_gat[(attn[0],attn[1])] = []
                        attention_train3[(attn[0],attn[1])].append(attn[2])
                        # attention_train3_gat[(attn[0],attn[1])].append(attn[3])
                    
                    case_numbers = data.case_number 
                    case_query = case_numbers[0]

                    out_temp = out[0]
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
                    sim_list = []
                    sim_list1 = []
                    sim_other = []
                    sim_3 = []
                    sims_all = F.cosine_similarity(out_temp.unsqueeze(0), out1, dim=1)
                    for i in range(len(first_indices)):
                        sim = sims_all[i]
                        if unique_cases[i] in label[case_query]:
                            sim_list.append([sim,unique_cases[i],label[case_query][unique_cases[i]]])
                            sim_list1.append(sim)
                            if label[case_query][unique_cases[i]] in [2,3]:
                                sim_3.append(sim)
                            else:
                                sim_other.append(sim)
                        else:
                            sim_other.append(sim)
                    j+=1
                    loss1 = 0

                    result[case_query]={}
                    sim_list.sort(key = lambda x:-x[0])
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
                    if len(sim_3)!=0 and len(sim_other)!=0:
                        loss1 = margin_ranking_loss(torch.stack(sim_3),torch.stack(sim_other))+ranking_loss_with_metrics(data_json,label_dic)

                    case_name = data.name
                    case_node_type = data.node_type

                    query_import =  [
                        idx for idx, (name, node_type, case_number) in enumerate(zip(case_name, case_node_type, case_numbers))
                        if case_number ==case_query and name == "important" and node_type == "virtual"
                    ][0]

                    query_unimport = [
                        idx for idx, (name, node_type, case_number) in enumerate(zip(case_name, case_node_type, case_numbers))
                        if case_number ==case_query and name == "unimportant" and node_type == "virtual"
                    ][0]

                    out_query_import = out[query_import]
                    out_query_unimport = out[query_unimport]
                    important_virtual_indices = []
                    for case in unique_cases:
                        for idx, (name, node_type, case_number) in enumerate(zip(case_name, case_node_type, case_numbers)):
                            if case_number == case and name == "important" and node_type == "virtual":
                                important_virtual_indices.append(idx)
                                break
                    import_node_features = torch.stack([out[i] for i in important_virtual_indices])
                    unimportant_virtual_indices = []
                    for case in unique_cases:
                        for idx, (name, node_type, case_number) in enumerate(zip(case_name, case_node_type, case_numbers)):
                            if case_number == case and name == "unimportant" and node_type == "virtual":
                                unimportant_virtual_indices.append(idx)
                                break
                    
                    unimport_node_features = torch.stack([out[i] for i in unimportant_virtual_indices])
                    sim_import = []
                    sim_unimport = []
                    sim_3_import = []
                    sim_other_import = []
                    sim_3_unimport = []
                    sim_other_unimport = []
                    sims_all1 =  F.cosine_similarity(out_query_import.unsqueeze(0),import_node_features,dim=1)
                    sims_all2 =  F.cosine_similarity(out_query_unimport.unsqueeze(0),unimport_node_features,dim=1)
                    for i in range(len(important_virtual_indices)):
                        sim = sims_all1[i]
                        sim_import.append(sim)
                        if unique_cases[i] in label[case_query]:
                            if label[case_query][unique_cases[i]] in [2,3]:
                                sim_3_import.append(sim)
                            else:
                                sim_other_import.append(sim)
                        else:
                            sim_other_import.append(sim)
                        sim = sims_all2[i]
                        sim_unimport.append(sim)
                        if unique_cases[i] in label[case_query]:
                            if label[case_query][unique_cases[i]] in [2,3]:
                                sim_3_unimport.append(sim)
                            else:
                                sim_other_unimport.append(sim)
                        else:
                            sim_other_unimport.append(sim)
                    if len(sim_3_unimport)!=0 and len(sim_other_unimport)!=0:
                        loss2 = margin_ranking_loss(torch.stack(sim_3_unimport),torch.stack(sim_other_unimport))
                    if len(sim_3_import)!=0 and len(sim_other_import)!=0:
                        loss3 = margin_ranking_loss(torch.stack(sim_3_import),torch.stack(sim_other_import))
                    import_score = torch.stack(sim_import)
                    unimport_score = torch.stack(sim_unimport)
                    labels = torch.ones(import_score.size(), device=import_score.device)  # [N, M]

                    # db_loss
                    loss_fn = nn.MarginRankingLoss(margin=1)
                    loss = loss_fn(import_score.flatten(), unimport_score.flatten(), labels.flatten())
                    if len(sim_3_import)!=0 and len(sim_other_import)!=0:
                        loss4 = loss1+loss3+loss2+loss
                        print("losstr:",loss,loss1,loss2,loss3,loss4)
                        loss_all_train.append(loss4)
                        loss4.backward()
                    else:
                        print("losstr:",loss4)
                        loss4 = loss
                        sim_list.sort(key = lambda x:-x[0])
                    optimizer.step()
                    torch.cuda.empty_cache()
                print("train_loss_all",torch.mean(torch.stack(loss_all_train)))
                print("val")
                del out, attention, attention2, attention3
                model.eval()
                result = {}
                
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
                        output = model(data)
                        out, attention, attention2, attention3 = output[0], output[1], output[2], output[3]

                        for attn in attention:
                            if (attn[0],attn[1]) not in attention_val:
                                attention_val[(attn[0],attn[1])] = []
                                # attention_train_gat[(attn[0],attn[1])] = []
                            attention_val[(attn[0],attn[1])].append(attn[2])
                            # attention_train_gat[(attn[0],attn[1])].append(attn[3])
                        for attn in attention2:
                            if (attn[0],attn[1]) not in attention_val2:
                                attention_val2[(attn[0],attn[1])] = []
                                # attention_train2_gat[(attn[0],attn[1])] = []
                            attention_val2[(attn[0],attn[1])].append(attn[2])
                            # attention_train2_gat[(attn[0],attn[1])].append(attn[3])
                        for attn in attention3:
                            if (attn[0],attn[1]) not in attention_val3:
                                attention_val3[(attn[0],attn[1])] = []
                                # attention_train3_gat[(attn[0],attn[1])] = []
                            attention_val3[(attn[0],attn[1])].append(attn[2])
                            # attention_train3_gat[(attn[0],attn[1])].append(attn[3])
                        
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
                        sim_list.sort(key = lambda x:-x[0])
                        result[case_query]={}
                        result_all[case_query] = {}
                        for item in sim_list:
                            result[case_query][item[1]] = item[0].item()
                            result_all[case_query][item[1]] = item[0].item()
                        torch.cuda.empty_cache()
                        with open("./experiment/"+exp_name+"/times"+str(n)+"/"+str(epoch)+"result.json","w",encoding='utf-8')as f:
                            json.dump(result, f, ensure_ascii = False, indent = 4)
                    
                    for attn in attention_train:
                        attention_train[attn] = [sum(attention_train[attn]) / len(attention_train[attn]), len(attention_train[attn]) ]
                    for attn in attention_val:
                        attention_val[attn] = [sum(attention_val[attn]) / len(attention_val[attn]) ,len(attention_val[attn])  ]
                    for attn in attention_train2:
                        attention_train2[attn] = [sum(attention_train2[attn]) / len(attention_train2[attn]), len(attention_train2[attn]) ]
                    for attn in attention_val2:
                        attention_val2[attn] = [sum(attention_val2[attn]) / len(attention_val2[attn]) ,len(attention_val2[attn])  ]
                    for attn in attention_train3:
                        attention_train3[attn] = [sum(attention_train3[attn]) / len(attention_train3[attn]), len(attention_train3[attn]) ]
                    for attn in attention_val3:
                        attention_val3[attn] = [sum(attention_val3[attn]) / len(attention_val3[attn]) ,len(attention_val3[attn])  ]
                    print(attention_train)
                    print(attention_val)
                    print(attention_train2)
                    print(attention_val2)
                    print(attention_train3)
                    print(attention_val3)   


                    print("test_loss_all", torch.mean(torch.stack(loss_all_test)))   
                    if torch.mean(torch.stack(loss_all_test)) < loss_min:
                        loss_min = torch.mean(torch.stack(loss_all_test))
                        torch.save(model, "./checkpoints/"+exp_name+".pth") 









    



    






    








