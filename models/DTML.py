import torch
import torch.nn as nn


class TimeAxisAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.tensor, rt_attn=False):
        # x: (D, W, L)
        o, (h, _) = self.lstm(x) # o: (D, W, H) / h: (1, D, H)
        score = torch.bmm(o, h.permute(1, 2, 0)) # (D, W, H) x (D, H, 1)
        tx_attn = torch.softmax(score, 1).squeeze(-1)  # (D, W)
        context = torch.bmm(tx_attn.unsqueeze(1), o).squeeze(1)  # (D, 1, W) x (D, W, H)
        normed_context = self.lnorm(context)
        if rt_attn:
            return normed_context, tx_attn
        else:
            return normed_context, None
        
class DataAxisAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, drop_rate=0.1):
        super().__init__()
        self.multi_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.ReLU(),
            nn.Linear(4*hidden_size, hidden_size)
        )
        self.lnorm1 = nn.LayerNorm(hidden_size)
        self.lnorm2 = nn.LayerNorm(hidden_size)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, hm: torch.tensor, rt_attn=False):
        # print(f'#DataAxisAttention hm {hm.shape}')
        # Forward Multi-head Attention
        residual = hm
        # hm_hat: (D, H), dx_attn: (D, D) 
        hm_hat, dx_attn = self.multi_attn(hm, hm, hm)
        # print(f'#DataAxisAttention hm {hm_hat.shape} dx_attn {dx_attn.shape}')
        hm_hat = self.lnorm1(residual + self.drop_out(hm_hat))
        
        # Forward FFN
        residual = hm_hat
        # hp: (D, H)
        hp = torch.tanh(hm + hm_hat + self.mlp(hm + hm_hat))
        hp = self.lnorm2(residual + self.drop_out(hp))

        if rt_attn:
            return hp, dx_attn
        else:
            return hp, None

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # input_size, hidden_size, num_layers, n_heads, beta=0.1, drop_rate=0.1

        self.beta = 0.1
        self.txattention = TimeAxisAttention(configs.enc_in, configs.d_model, configs.e_layers)
        self.dxattention = DataAxisAttention(configs.d_model, configs.n_heads, configs.dropout)
        self.linear = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, stocks, rt_attn=False):
        # stocks: (B, L, D)
        # B: number of stocks
        # L: length of observations
        # D: number of features

        # Time-Axis Attention
        # c_stocks: (D, H) / tx_attn_stocks: (D, W)
        c_stocks, tx_attn_stocks = self.txattention(stocks, rt_attn=rt_attn)
        
        # Context Aggregation
        # Multi-level Context
        # hm: (D, H)
        hm = c_stocks

        # Data-Axis Attention
        # hp: (D, H) / dx_attn: (D, D)
        hp, dx_attn_stocks = self.dxattention(hm, rt_attn=rt_attn)
        # output: (D, T)
        output = self.linear(hp).unsqueeze(2)

        return output

    # def forward(self, stocks, index, rt_attn=False):
    #     # stocks: (W, D, L) for a single time stamp
    #     # index: (W, 1, L) for a single time stamp
    #     # W: length of observations
    #     # D: number of stocks
    #     # L: number of features
        
    #     # Time-Axis Attention
    #     # c_stocks: (D, H) / tx_attn_stocks: (D, W)
    #     c_stocks, tx_attn_stocks = self.txattention(stocks.transpose(1, 0), rt_attn=rt_attn)
    #     # c_index: (1, H) / tx_attn_index: (1, W)
    #     c_index, tx_attn_index = self.txattention(index.transpose(1, 0), rt_attn=rt_attn)
        
    #     # Context Aggregation
    #     # Multi-level Context
    #     # hm: (D, H)
    #     hm = c_stocks + self.beta * c_index
    #     # The Effect of Global Contexts
    #     # effect: (D, D)
    #     effect = c_stocks.mm(c_stocks.transpose(0, 1)) + \
    #         self.beta * c_index.mm(c_stocks.transpose(1, 0)) + \
    #         self.beta**2 * torch.mm(c_index, c_index.transpose(0, 1)) 

    #     # Data-Axis Attention
    #     # hp: (D, H) / dx_attn: (D, D)
    #     hp, dx_attn_stocks = self.dxattention(hm, rt_attn=rt_attn)
    #     # output: (D, 1)
    #     output = self.linear(hp)

    #     return {
    #         'output': output,
    #         'tx_attn_stocks': tx_attn_stocks,
    #         'tx_attn_index': tx_attn_index,
    #         'dx_attn_stocks': dx_attn_stocks,
    #         'effect': effect
    #     }