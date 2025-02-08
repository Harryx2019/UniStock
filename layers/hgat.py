
import torch
from torch import nn
from torch_geometric.nn import GATConv, HypergraphConv
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from layers.Transformer_EncDec import EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class DyHGAT(nn.Module):
    """
    Dynamic Hypergraph Attention Network
    Paper: Stock trend prediction based on dynamic hypergraph spatio-temporal network
    Method: 4.2.1. Dynamic hypergraph construction module
    Function: 对输入的股票特征, 先利用GAT计算相关性, 再利用softmax构建超边, 最后利用HGAT更新股票特征
    """
    def __init__(self, in_channels, out_channels, heads, dropout, num_hyperedges, 
                 quantile=0.9, norm='layernorm', output_hyergraph=False):
        super().__init__()
        # Encoder Layer
        # 2024-1-5 原本使用GAT更新两两相连的节点编码, 但是显存占用非常大(因为两两相连的图太大), 因此考虑用一层自注意力层完成
        self.encoder_layer = EncoderLayer(
                                AttentionLayer(
                                    FullAttention(mask_flag=False,
                                                  attention_dropout=dropout,
                                                  output_attention=False),
                                    out_channels,
                                    heads),
                                d_model=out_channels,
                                dropout=dropout,
                                activation='gelu',
                                norm='LayerNorm'
                            )
        
        # Hypergraph construction
        self.fc_hypergraph = nn.Linear(out_channels, num_hyperedges)
        self.num_hyperedges = num_hyperedges
        self.quantile = quantile
        
        # Hypergraph convolution layer
        self.hgat1 = HypergraphConv(out_channels, out_channels, use_attention=True, heads=heads, concat=False, negative_slope=0.2, dropout=dropout, bias=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm = norm
        self.output_hyergraph = output_hyergraph
        if norm == 'batchnorm':
            self.bn1 = nn.BatchNorm1d(out_channels)
        elif norm == 'layernorm':
            self.bn1 = nn.LayerNorm(out_channels)
        
        self.hgat2 = HypergraphConv(out_channels, out_channels, use_attention=True, heads=1, concat=False, negative_slope=0.2, dropout=dropout, bias=True)
        self.dropout2 = nn.Dropout(dropout)
        if norm == 'batchnorm':
            self.bn2 = nn.BatchNorm1d(out_channels)
        elif norm == 'layernorm':
            self.bn2 = nn.LayerNorm(out_channels)

        self.linear = nn.Linear(out_channels, in_channels)
        self.dropout3 = nn.Dropout(dropout)
        if norm == 'batchnorm':
            self.bn3 = nn.BatchNorm1d(out_channels)
        elif norm == 'layernorm':
            self.bn3 = nn.LayerNorm(out_channels)

    def forward(self, x):
        """
            x [S, D]: S: num_stocks(in parallel), D: d_model
        """
        src = x
        n = x.shape[0] # num_stock(in parallel)
        # 25/1/4 由于使用BatchNorm, 导致如果输入的batchsize为1的话,会报错
        if n == 1 and self.norm == 'batchnorm':
            print(f'#DyHGAT x shape {x.shape}')
            return x, None
        
        # Step1.通过自注意力机制更新股票节点特征
        x, attn = self.encoder_layer(x.unsqueeze(0))
        x = x.squeeze(0)
        
        # Step3.构建超图
        # Step3.1 利用softmax进行权重计算
        hypergraph_adj = self.fc_hypergraph(x)               
        hypergraph_adj = torch.softmax(hypergraph_adj, dim=0)       # [S, num_hyperedges]
        threshold = torch.quantile(hypergraph_adj, self.quantile)   # 24-12/31 将计算的分位数作为超边的筛选依据
        
        # Step3.2 根据阈值对超边进行筛选
        hypergraph_adj = torch.where(hypergraph_adj >= threshold, torch.ones_like(hypergraph_adj), torch.zeros_like(hypergraph_adj))

        # 计算图的稠密度
        # sum_of_hyperedges = torch.sum(hypergraph_adj)
        # print(f"The density of the hypergraph: {sum_of_hyperedges.item() / (hypergraph_adj.shape[0] * hypergraph_adj.shape[1])}")
        
        hypergraph_adj_sparse = hypergraph_adj.nonzero().t()
        assert hypergraph_adj_sparse.shape[0] == 2, f"hypergraph_adj_sparse should have shape [2, E], but got {hypergraph_adj.shape}"
        
        # Step3.3 利用超图更新股票节点特征
        hyperedge_weight = torch.ones(self.num_hyperedges, device=src.device)
        hyperedge_attr = torch.matmul(hypergraph_adj.T, src)          # [num_hyperedges, D]
        
        x1 = self.hgat1(src, hypergraph_adj_sparse, hyperedge_weight, hyperedge_attr)
        # Add & Norm
        x = src + self.dropout1(x1)
        x = self.bn1(x)                                             # x: [S, D]
        
        hyperedge_attr = torch.matmul(hypergraph_adj.T, x)          # [num_hyperedges, D]
        x2 = self.hgat2(x, hypergraph_adj_sparse, hyperedge_weight, hyperedge_attr)
        # Add & Norm
        x = x + self.dropout2(x2)
        x = self.bn2(x)                                             # x: [S, D]

        x = F.leaky_relu(self.linear(x), 0.2)
        x = src + self.dropout3(x)
        x = self.bn3(x)                                             # x: [S, D]

        if self.output_hyergraph:
            return x, hypergraph_adj
        else:
            return x, None
    

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class HGAT(nn.Module):
    def __init__(self, d_in, d_model, dropout):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model

        self.linear1 = nn.Linear(d_in, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(d_model)
        
        self.hatt1 = HypergraphConv(d_model, d_model, use_attention=True, heads=4, concat=False, negative_slope=0.2, dropout=0.2, bias=True)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(d_model)

        self.hatt2 = HypergraphConv(d_model, d_model, use_attention=True, heads=1, concat=False, negative_slope=0.2, dropout=0.2, bias=True)
        self.dropout3 = nn.Dropout(dropout)
        self.bn3 = nn.BatchNorm1d(d_model)

        self.linear2 = nn.Linear(d_model, d_in)
        self.dropout4 = nn.Dropout(dropout)
        self.bn4 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))

    def forward(self, x, hyperedge_index, device):                      # x: [bs x patch_num x d_model] hyperedge_index: [2 x hyperedge_nums]
        src = x                                                                                   # src: [bs x patch_num x d_model]
        self.device = device

        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))                                 # x: [bs x patch_num * d_model]
        x = F.leaky_relu(self.linear1(x),0.2)                                                     
        x = self.dropout1(x)
        x = self.bn1(x)                                                                           # x: [bs x d_model]

        num_nodes = x.shape[0]
        num_edges = hyperedge_index[1].max().item() + 1

        a = to_dense_adj(hyperedge_index)[0].to(self.device)                                      # a: [bs x num_edges]
        if num_nodes > num_edges:
            a = a[:,:num_edges]
        else:
            a = a[:num_nodes]
        hyperedge_weight = torch.ones(num_edges).to(self.device)                                  # hyperedge_weight: [num_edges]
        hyperedge_attr = torch.matmul(a.T, x)                                                     # hyperedge_attr: [num_edges x d_model]

        x2 = self.hatt1(x, hyperedge_index, hyperedge_weight, hyperedge_attr)                     # x2: [bs x d_model]
        # Add & Norm
        x = x + self.dropout2(x2) # Add: residual connection with residual dropout
        x = self.bn2(x)

        hyperedge_attr = torch.matmul(a.T, x)                                                     # hyperedge_attr: [num_edges x d_model]
        x2 = self.hatt2(x, hyperedge_index, hyperedge_weight, hyperedge_attr)                     # x2: [bs x d_model]
        # Add & Norm
        x = x + self.dropout3(x2) # Add: residual connection with residual dropout
        x = self.bn3(x)
        
        x = F.leaky_relu(self.linear2(x), 0.2)                                                    # x: [bs x patch_num * d_model]
        x = torch.reshape(x, (x.shape[0], -1, self.d_model))                                      # x: [bs x patch_num x d_model]
        # Add & Norm
        x = src + self.dropout4(x) # Add: residual connection with residual dropout
        x = self.bn4(x)                                                                           # x: [bs x patch_num x d_model]
        return x