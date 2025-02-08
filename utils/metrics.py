import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / (true + 1e-8)))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / (true + 1e-8)))


def metric(pred, true):
    mae = MAE(pred, true)   # Mean Absolute Error
    mse = MSE(pred, true)   # Mean Squared Error
    rmse = RMSE(pred, true) # Root Mean Squared Error
    mape = MAPE(pred, true) # Mean Absolute Percentage Error
    mspe = MSPE(pred, true) # Mean Squared Percentage Error

    return mae, mse, rmse, mape, mspe


# loss function
def weighted_mse_loss(input, target, weight):
    '''
        input shape     [stock_num, pred_len]
        target shape    [stock_num, pred_len]
        weight shape    [stock_num, pred_len]
    '''
    # print(f'input {input.shape}')
    # print(f'target {target.shape}')
    # print(f'weight {weight.shape}')
    return torch.mean(weight * (input - target) ** 2)

def mse_rank_loss(i, pred, base_price, ground_truth, alpha, device):
    '''
        i: batch index
        pred: model's predicted close                           [num_stock, pred_len, 1]
        base_price: true close price at t - 1                   [num_stock, 1] 【包含nan值】
        ground_truth: true return tatio in [t: t + pred_len]    [num_stock, pred_len] 【包含nan值】
        alpha: hyerparameter
    '''
    # print(f'mse_rank_loss pred {pred.shape} base_price {base_price.shape}'
    #       f'ground_truth {ground_truth.shape} mask {mask.shape} alpha={alpha} device={device}')
    # 24/12-31 将以下逻辑更改到exp_stock中
    # mask = ~torch.isnan(base_price.squeeze()) # [num_stock] 当日有效交易股票数
    # pred = pred[mask]
    # base_price = base_price[mask]
    # ground_truth = ground_truth[mask]
    
    pred = pred.squeeze(-1)     # [num_stock, pred_len]
    num_stocks = pred.shape[0]
    pred_len = pred.shape[1]
    
    return_ratio = torch.div((pred - base_price), base_price)       # [num_stock, pred_len]
    
    # mse loss
    # 24-12/26 训练集回归损失出现nan
    mask = ~torch.isnan(ground_truth) & ~torch.isnan(return_ratio)   # [num_stock, pred_len] 有效样本数
    valid_elements = mask.sum().item()
    if valid_elements > 0:
        reg_loss = torch.mean((return_ratio[mask] - ground_truth[mask]) ** 2)
    else:
        print(f"i{i} t{t+1} No valid elements for mse loss, skipping this iteration.")
        reg_loss = torch.tensor(0.0, device='cuda')
    
    # rank loss
    all_ones = torch.ones(num_stocks, 1).to(device)
    rank_loss = torch.tensor(0.0, device='cuda')  # 初始化为一个 GPU 张量
    count = 0
    for t in range(pred_len):
        pre_pw_dif =  (torch.matmul(return_ratio[:,t:t+1], torch.transpose(all_ones, 0, 1))
                    - torch.matmul(all_ones, torch.transpose(return_ratio[:,t:t+1], 0, 1)))
        gt_pw_dif = (torch.matmul(all_ones, torch.transpose(ground_truth[:,t:t+1], 0, 1)) 
                    - torch.matmul(ground_truth[:,t:t+1], torch.transpose(all_ones, 0, 1)))
        
        mask_pw = ~torch.isnan(pre_pw_dif) & ~torch.isnan(gt_pw_dif)  # [num_stock, num_stock]
        valid_elements = mask_pw.sum().item()
        if valid_elements > 0:
            count += 1
            rank_loss += torch.mean(F.relu(pre_pw_dif[mask_pw] * gt_pw_dif[mask_pw]))
        else:
            print(f"i{i} t{t+1} No valid elements for mask_pw, skipping this iteration.")
    if count > 0:
        rank_loss /= count
    
    loss = reg_loss + alpha * rank_loss
    del gt_pw_dif, pre_pw_dif, mask, all_ones, mask_pw
    return loss, reg_loss, rank_loss, return_ratio
    

def autoregression_mse_loss(preds, target):
    '''
        preds shape     [stock_num, pred_len(=seq_len), n_vars]
        target shape    [stock_num, pred_len(=seq_len), n_vars]
    '''
    return torch.mean((preds - target) ** 2)


def reconstruction_mse_loss(preds, target, mask):
    """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        mask:    [bs x num_patch x n_vars]
    """
    loss = (preds - target) ** 2
    loss = loss.mean(dim = -1)
    loss = (loss * mask).sum() / mask.sum()
    return loss

def get_top_k_stocks(data, mask, t, i, k):
    """获取第 t 天T + i的预测或真实前 k 名股票索引，跳过 mask 中为 0 的部分。"""
    ranked_indices = np.argsort(data[:, t, i])[::-1]  # 降序排列索引[ic为正]
    # ranked_indices = np.argsort(data[:, t, i])  # 升序排列索引[ic为负]
    top_k = [idx for idx in ranked_indices if mask[idx, t, i] == True][:k]
    return top_k

def accuracy(pred, true):
    '''
        计算预测收益率和真实收益率的准确率, 即模型预测和真实涨跌一致的百分比
    '''
    # 判断涨跌方向
    pred_direction = (pred >= 0).astype(int)
    true_direction = (true >= 0).astype(int)
    
    # 比较方向是否一致
    correct_predictions = (pred_direction == true_direction).sum()
    
    # 计算准确率
    accuracy = correct_predictions / len(pred_direction)
    return accuracy

# backtest
def backtest(prediction, ground_truth, path, market_name, k_list=[1, 5, 10, 15, 20]):
    '''
        prediction:     [num_stock, test_days, output_len]【包含nan值】
        ground_truth:   [num_stock, test_days, output_len]【包含nan值】
    '''
    assert ground_truth.shape == prediction.shape, 'shape mis-match between prediction and ground_truth'
    
    mask = ~np.isnan(prediction) & ~np.isnan(ground_truth) # [num_stock, test_days, output_len]
    mae, mse, rmse, mape, mspe = metric(prediction[mask], ground_truth[mask])
    acc = accuracy(prediction[mask], ground_truth[mask])
    
    num_stock, test_days, output_len = prediction.shape
    performance = []
    # 年化收益率, 年化夏普比率, 收益率序列, 选股列表, rank_ic序列
    arr_list, sharpe_list, irr_list, selected_stock_list, rank_ic_list = [], [], [], [], []
    # benchmark 年化收益率, 年化夏普比率, 收益率序列
    bm_arr_list, bm_sharpe_list, bm_irr_list = [], [], []
    
    # i循环预测长度 T + i[持有周期]
    for i in range(output_len):
        arr_list_i, sharpe_list_i, irr_list_i, selected_stock_list_i, rank_ic_list_i = [], [], [], [], []

        # benchmark 累积收益率, 每日收益率, 累积收益率序列
        bm_bt_long, bm_daily_returns, bm_irr = 1.0, [], []

        # k循环持股数量 TopK
        for k in k_list:
            performance_k = {'T': i + 1, 'K': k}

            # 累积收益率, 每日收益率, 累积收益率序列, 选股列表
            bt_long, daily_returns, irr, selected_stock = 1.0, [], [], []

            # t 循环测试天数
            for t in range(test_days):
                # 预测前k名排序
                pred_topk = get_top_k_stocks(prediction, mask, t, i, k)
                selected_stock.append(pred_topk)
                pred_topk = np.array(pred_topk)

                # back testing on top k
                real_ret_rat_topk = ground_truth[pred_topk, t, i]
                # 24-12-14 打印收益率大于阈值的情况
                if market_name == 'A_share':
                    threshold = 0.1 # A股0.1
                elif market_name == 'S&P500':
                    threshold = 0.3 # 美股0.3
                else:
                    raise NotImplementedError

                indices = np.where(abs(real_ret_rat_topk) >= threshold)
                if indices[0].size != 0:
                    # print(f'#Warning T + {i + 1} Top {k} {t}-th stock {pred_topk[indices]} return ratio {real_ret_rat_topk[indices]}')
                    pass
                
                # 24-11-13 重要！！考虑到回测仓位, 每日持仓比例为1/peroid(调仓周期)
                real_ret_rat_topk = np.mean(real_ret_rat_topk) / (i + 1)
                bt_long *= (1 + real_ret_rat_topk) # 累积收益率(复利)
                daily_returns.append(real_ret_rat_topk)
                irr.append(bt_long)

                if k == k_list[0]:
                    valid_indices = mask[:, t, i] == True
                    if np.sum(valid_indices) > 1:  # 至少需要两个有效元素
                        # 计算 截面 Rank IC（使用 Spearman 相关系数）
                        rank_ic, _ = spearmanr(
                            prediction[valid_indices, t, i],
                            ground_truth[valid_indices, t, i]
                        )
                        rank_ic_list_i.append(rank_ic)

                        # 计算 benchmark
                        real_ret_rat_bm = np.mean(ground_truth[valid_indices, t, i]) / (i + 1)
                        bm_bt_long *= (1 + real_ret_rat_bm) # 累积收益率(复利)
                        bm_daily_returns.append(real_ret_rat_bm)
                        bm_irr.append(bm_bt_long)
                    else:
                        raise ValueError(f'Backtest value error! {t}-th T+{i+1} valid_indices < 1')
            
            # 计算年化收益率和夏普比率
            performance_k['btl'] = bt_long - 1
            arr = (bt_long ** (252 / test_days)) - 1 # 年化收益率
            performance_k['Ann return'] = arr
            daily_returns = np.array(daily_returns)
            sharpe = (np.mean(daily_returns)/(np.std(daily_returns)+ 1e-8)) * 15.87 # 年化夏普比率
            performance_k['Ann sharpe'] = sharpe

            if k == k_list[0]:
                # Rank IC 和 Rank ICIR 计算
                rank_ic_list_i = np.array(rank_ic_list_i)
                rank_ic_mean = np.mean(rank_ic_list_i)
                rank_ic_std = np.std(rank_ic_list_i)
                rank_icir = rank_ic_mean / (rank_ic_std + 1e-8)  # 防止除以 0

                bm_arr = (bm_bt_long ** (252 / test_days)) - 1 # 年化收益率
                bm_daily_returns = np.array(bm_daily_returns)
                bm_sharpe = (np.mean(bm_daily_returns)/(np.std(bm_daily_returns)+ 1e-8)) * 15.87 # 年化夏普比率

            performance_k['rank_ic'] = rank_ic_mean
            performance_k['rank_icir'] = rank_icir

            performance_k['Benchmark Ann return'] = bm_arr
            performance_k['Benchmark Ann sharpe'] = bm_sharpe

            performance.append(performance_k)
            
            arr_list_i.append(arr)
            sharpe_list_i.append(sharpe)
            irr_list_i.append(irr)
            selected_stock_list_i.append(selected_stock)

        arr_list.append(arr_list_i)
        sharpe_list.append(sharpe_list_i)
        irr_list.append(irr_list_i)
        selected_stock_list.append(selected_stock_list_i)
        
        rank_ic_list.append(rank_ic_list_i)

        bm_arr_list.append(bm_arr)
        bm_sharpe_list.append(bm_sharpe)
        bm_irr_list.append(bm_irr)


    arr_list = np.array(arr_list)                                           # [output_len, k]
    sharpe_list = np.array(sharpe_list)                                     # [output_len, k]
    irr_list = np.array(irr_list)                                           # [output_len, k, test_days]
    selected_stock_list = np.array(selected_stock_list, dtype=object)       # [output_len, k, test_days, k]
    rank_ic_list = np.array(rank_ic_list)                                   # [output_len, test_days]
    bm_arr_list = np.array(bm_arr_list)                                     # [output_len]
    bm_sharpe_list = np.array(bm_sharpe_list)                               # [output_len]
    bm_irr_list = np.array(bm_irr_list)                                     # [output_len, test_days]
    
    print(f'\nBacktest done')
    print(f'arr {arr_list.shape} sharpe {sharpe_list.shape} irr {irr_list.shape} '
         f'selected_stock {selected_stock_list.shape} rank_ic {rank_ic_list.shape} '
         f'bm_arr {bm_arr_list.shape} bm_sharpe {bm_sharpe_list.shape} bm_irr {bm_irr_list.shape}\n')
    
    performance = pd.DataFrame(performance)
    performance.to_csv(path + '/' + 'performance.csv', index=False)
    # np.save(path + '/' + 'arr.npy', arr_list)
    # np.save(path + '/' + 'sharpe.npy', sharpe_list)
    np.save(path + '/' + 'irr.npy', irr_list)
    # np.save(path + '/' + 'selected_stock.npy', selected_stock_list)
    np.save(path + '/' + 'ic.npy', rank_ic_list)
    # np.save(path + '/' + 'bm_arr.npy', bm_arr_list)
    # np.save(path + '/' + 'bm_sharpe.npy', bm_sharpe_list)
    np.save(path + '/' + 'bm_irr.npy', bm_irr_list)
    
    return mae, mse, rmse, mape, mspe, acc, performance