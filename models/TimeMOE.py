import torch

class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.prediction_length = 5
        self.context_length = 672
        

    def forward(self, normed_seqs):
        """
        预测给定上下文序列的未来值。
        
        参数:
        normed_seqs (torch.Tensor): 输入的归一化序列，形状为 [batch_size, context_length]
        
        返回:
        torch.Tensor: 预测结果，形状为 [batch_size, prediction_length]
        """
        normed_seqs = normed_seqs.squeeze(-1) 
        output = self.model.generate(normed_seqs, max_new_tokens=self.prediction_length)
        # 获取预测部分（即最后 prediction_length 的部分）
        normed_predictions = output[:, -self.prediction_length:]  # 形状是 [batch_size, prediction_length]
        normed_predictions = normed_predictions.unsqueeze(-1)
        return normed_predictions