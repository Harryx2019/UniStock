import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, MaskPatchEmbedding
from layers.hgat import DyHGAT

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x                            # x: [bs x nvars x target_window]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, head_dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x num_patch x d_model]
        output: tensor [bs x nvars x num_patch x patch_len]
        """
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        return x



class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.graph = configs.graph
        print(f'self.graph:{self.graph}')

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if self.seq_len % self.patch_len != 0:
            padding = 0 # TODO: padding大小的确定
            raise ValueError(f'Error!! seq_len % patch_len != 0')
        else:
            padding = 0

        # patching and embedding
        if self.task_name == 'pretrain':
            # 固定mask比率为0.4
            print('model CI-STHPAN MaskPatchEmbedding')
            self.patch_embedding = MaskPatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout, 0.4)
        elif self.task_name in ['finetune', 'supervised']:
            print('model CI-STHPAN PatchEmbedding')
            self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, padding, configs.dropout)
        else:
            raise NotImplementedError
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False,
                                      factor=configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False), # output_attention 决定是否输出attention score in FullAttention
                        configs.d_model,
                        configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation, # 只有relu 和 gelu 两个选项
                    norm='LayerNorm' # TODO: PatchTST原模型为BatchNorm
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )
        
        self.patch_num = (self.seq_len - self.patch_len) // self.stride + 1
        print('number of patches:', self.patch_num)

        # HGAT
        if self.graph:
            self.hgat = DyHGAT(configs.d_model, configs.d_model, configs.n_heads, configs.dropout,
                              configs.num_hyperedges, configs.hyperedges_quantile, 'layernorm', output_hyergraph=False)
        
        # Head
        # 25-1-12 微调和预训练共享一个预测头
        self.head = PretrainHead(configs.d_model, self.patch_len, configs.dropout)

        # if self.task_name == 'pretrain':
        #     print('model CI-STHPAN PretrainHead')
        #     self.head = PretrainHead(configs.d_model, self.patch_len, configs.dropout)
        # elif self.task_name == 'finetune':
        #     print('model CI-STHPAN FlattenHead')
        #     self.head_nf = configs.d_model * self.patch_num  # Flatten
        #     self.head = FlattenHead(self.head_nf, self.pred_len, configs.dropout)
        # else:
        #     raise NotImplementedError
    

    def pretrain(self, x_enc, hyperedge_index=None, device='cpu'):
        # x_enc: [B, L, D]  hyperedge_index: [D, B, B] B: stock_num
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev                                                      # x_enc: [B, L, D]

        # do mask patching and embedding
        # x 为原始patch后的输入 [bs x num_patch x n_vars x patch_len]
        # enc_out 为mask后的输入 [bs * n_vars x num_patch x d_model]
        # mask 为被mask的patch [bs x num_patch x n_vars]
        x, enc_out, mask, n_vars = self.patch_embedding(x_enc)              # enc_out: [B * D, N, d]
        B, N, D, P = x.shape

        # Encoder
        enc_out, attns = self.encoder(enc_out)                              # enc_out: [B * D, N, d]
        
        # Decoder
        enc_out = torch.reshape(enc_out, (B, D, N, enc_out.shape[-1]))      # enc_out: [B, D, N, d]
        dec_out = self.head(enc_out)                                        # dec_out: [B, D, N, P]
        dec_out = dec_out.permute(0, 2, 3, 1)                               # dec_out: [B, N, P, D]
        dec_out = torch.reshape(dec_out, (B, N * P, D))                     # dec_out: [B, N*P, D]
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, N*P, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, N*P, 1))

        dec_out = torch.reshape(dec_out, (B, N, P, D))    # dec_out: [B, N, P, D]
        dec_out = dec_out.permute(0, 1, 3, 2)             # dec_out: [B, N, D, P]
        return x, dec_out, mask


    def finetune(self, x_enc, hyperedge_index=None, device='cpu'):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # [B, L, D]
        B, L, D = x_enc.shape

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)                                      # x_enc: [B, D, L]
        enc_out, n_vars = self.patch_embedding(x_enc)                       # enc_out: [B * D, N, d]
        N = enc_out.shape[1]

        # Encoder
        enc_out, attns = self.encoder(enc_out)                              # enc_out: [B * D, N, d]

        # HGAT
        if self.graph:
            # 静态超图构建
            # assert hyperedge_index != None
            # enc_out = self.hgat(enc_out, hyperedge_index[0], device)            # enc_out: [B * D, N, d]
            
            # 动态超图 逐通道逐token沿着batch维度进行
            enc_out = enc_out.reshape(B, D, N, -1)
            new_enc_out = enc_out.clone()
            for i in range(D):
                for j in range(N):
                    new_enc_out[:, i, j, :], hypergraph_adj = self.hgat(enc_out[:, i, j, :])
                    # TODO: 对hypergraph_adj进行可视化
            enc_out = new_enc_out.reshape(B * D, N, -1)
        
        enc_out = torch.reshape(enc_out, (B, D, N, -1))                      # enc_out: [B, D, N, d]
        # enc_out = enc_out.permute(0, 1, 3, 2)                                # enc_out: [B, D, d, N]
        
        # Decoder
        dec_out = self.head(enc_out)                                        # dec_out: [B, D, N, P]
        dec_out = dec_out.reshape(B, D, -1)                                 # dec_out: [B, D, N*P]
        dec_out = dec_out.permute(0, 2, 1)                                  # dec_out: [B, N*P, D]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, N * self.patch_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, N * self.patch_len, 1))
        return dec_out, attns                                                      # dec_out: [B, T, D]


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, hyperedge_index=None, device='cpu'):
        # x_enc: [bs x seq_len x nvars] bs = stock_num
        if self.task_name in ['finetune', 'supervised']:
            # [bs x (patch_num * patch_len) x nvars]
            dec_out, attns = self.finetune(x_enc, hyperedge_index, device)
            return dec_out, attns
        elif self.task_name == 'pretrain':
            # x       [bs x num_patch x n_vars x patch_len]
            # dec_out [bs x num_patch x n_vars x patch_len]
            # mask    [bs x num_patch x n_vars]
            x, dec_out, mask = self.pretrain(x_enc, hyperedge_index, device)
            return x, dec_out, mask
        return None
    
