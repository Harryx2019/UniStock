import torch
from torch import nn
from layers.Transformer_EncDec import DecoderOnly, DecoderOnlyLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding
from layers.hgat import DyHGAT


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2402.02368
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.graph = configs.graph
        print(f'self.graph:{self.graph}')

        self.input_token_len = configs.input_token_len
        self.embedding = nn.Linear(self.input_token_len, configs.d_model, bias=False)
        self.position_embedding = PositionalEmbedding(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.blocks = DecoderOnly(
            [
                DecoderOnlyLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=True), 
                        configs.d_model, 
                        configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        if self.graph:
            self.hgat = DyHGAT(configs.d_model, configs.d_model, configs.n_heads, configs.dropout,
                              configs.num_hyperedges, configs.hyperedges_quantile, 'layernorm', output_hyergraph=False)
        
        self.head = nn.Linear(configs.d_model, configs.output_token_len)
        self.use_norm = configs.use_norm # RevIN
    
    def forecast(self, x):
        # 对每个样本每个特征沿着时间轴进行标准化
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        # [B, L, C]
        B, _, C = x.shape
        # [B, C, L]
        x = x.permute(0, 2, 1)
        # [B, C, N, P]
        x = x.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x.shape[2]
        # [B * C, N, P]
        x = x.reshape(B * C, N, -1)
        # [B * C, N, D]
        embed_out = self.embedding(x) + self.position_embedding(x)
        embed_out = self.dropout(embed_out)
        embed_out, attns = self.blocks(embed_out)

        if self.task_name == 'finetune' and self.graph:
            # 逐通道对最后一个token沿着batch维度进行动态超图
            embed_out = embed_out.reshape(B, C, N, -1)
            new_embed_out = embed_out.clone()  # 创建一个新的张量来存储结果
            for i in range(C):
                for j in range(N):
                    # 25-1-6 验证过用N个token建模的效果优于只用最后一个token的效果
                    new_embed_out[:, i, j, :], hypergraph_adj = self.hgat(embed_out[:, i, j, :])
                # TODO: 对hypergraph_adj进行可视化
            embed_out = new_embed_out.reshape(B * C, N, -1)
        
        # [B * C, N, P]
        dec_out = self.head(embed_out)
        # [B, C, L]
        dec_out = dec_out.reshape(B, C, -1)
        # [B, L, C]
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * stdev + means
        return dec_out, attns
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, hyperedge_index=None, device='cpu'):
        return self.forecast(x_enc)
