import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import time
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class UTSD_Npy(Dataset):
    def __init__(self, args, market_name, root_path, data_path, flag, 
                 input_token_len, output_token_len, autoregressive,
                 size, features, target, timeenc, freq,
                 scale, stride=1, split=0.9):
        self.args = args
        self.market_name = market_name # 股票市场
        
        self.seq_len = size[0]
        self.input_token_len = input_token_len
        self.output_token_len = output_token_len
        self.context_len = self.seq_len + self.output_token_len
        print(f'Context length: {self.context_len}')
        
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = os.path.join(root_path, data_path)
        print("Load data from: ", self.root_path)

        self.scale = scale      # 数据标准化
        self.split = split      # 训练集划分
        self.stride = stride    # 数据采样
        self.data_list = []     # 数据集列表, 一个数据集一个numpy
        self.n_window_list = [] # 数据集下标, n_timepoint * n_var[累加]
        self.__confirm_data__()

    def __confirm_data__(self):
        for root, dirs, files in tqdm(os.walk(self.root_path)):
            for file in files:
                if file.endswith('.npy'):
                    dataset_path = os.path.join(root, file)

                    if self.market_name in ['Stock', 'Stock_maxscale', 'Stock_only', 'Stock_only_maxscale']:
                        if ('A_share' not in dataset_path) and ('S&P500' not in dataset_path):
                            continue
                    elif self.market_name in ['A_share', 'A_share_maxscale', 'A_share_only', 'A_share_only_maxscale']:
                        if 'A_share' not in dataset_path:
                            continue
                    elif self.market_name in ['S&P500', 'S&P500_maxscale', 'S&P500_only', 'S&P500_only_maxscale']:
                        if 'S&P500' not in dataset_path:
                            continue
                    elif self.market_name == 'UTSD':
                        if ('A_share' in dataset_path) or ('S&P500' in dataset_path):
                            continue
                    else:
                        raise NotImplementedError
                    # print(f'load data from: {dataset_path}')
                    data = np.load(dataset_path)
                    
                    rows_with_nan = np.isnan(data).any(axis=1)
                    num_rows_with_nan = np.sum(rows_with_nan)
                    if num_rows_with_nan != 0:
                        # 均值填充nan值
                        # print(f'Error! data from {dataset_path} has {num_rows_with_nan} rows nan !')
                        imputer = SimpleImputer(strategy='mean')
                        data = imputer.fit_transform(data)
                    
                    num_train = int(len(data) * self.split)
                    num_test = int(len(data) * (1 - self.split) / 2)
                    num_vali = len(data) - num_train - num_test
                    if num_train < self.context_len:
                        # 跳过数据长度不足模型输入长度的样本
                        # print(f'skip data length isn\'t enough {dataset_path}')
                        continue
                    border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
                    border2s = [num_train, num_train + num_vali, len(data)]

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]

                    if self.scale:
                        train_data = data[border1s[0]:border2s[0]]
                        
                        if 'maxscale' in self.market_name:
                            if ('A_share' in dataset_path) or ('S&P500' in dataset_path):
                                # 股票数据最大值标准化
                                print(f'max scale {dataset_path}')
                                max_value = train_data.max()
                                if max_value == 0:
                                    raise ValueError(f'{dataset_path} max_val = 0!!')
                                data = data / max_value
                            else:
                                raise NotImplementedError
                        else:
                            scaler = StandardScaler()
                            scaler.fit(train_data)
                            data = scaler.transform(data)
                    else:
                        raise NotImplementedError

                    data = data[border1:border2]
                    n_timepoint = (len(data) - self.context_len) // self.stride + 1
                    n_var = data.shape[1]
                    self.data_list.append(data)

                    n_window = n_timepoint * n_var
                    self.n_window_list.append(n_window if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
        print("Total number of windows in merged dataset: ", self.n_window_list[-1])

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - self.n_window_list[dataset_index - 1] if dataset_index > 0 else index
        n_timepoint = (len(self.data_list[dataset_index]) - self.context_len) // self.stride + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        # input_token_len = output_token_len
        r_begin = s_begin + self.input_token_len
        r_end = s_end + self.output_token_len

        seq_x = self.data_list[dataset_index][s_begin:s_end, c_begin:c_begin + 1]
        seq_y = self.data_list[dataset_index][r_begin:r_end, c_begin:c_begin + 1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_window_list[-1]
    

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test data_UTSD_Npy')

    parser.add_argument('--batch_size', type=int, default=2048, help='batch size of train input data')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    # data loader
    parser.add_argument('--data', type=str, default='Utsd_Npy', help='dataset type')
    parser.add_argument('--market_name', type=str, default='A_share', help='stock market name')
    parser.add_argument('--root_path', type=str, default='/home/xiahongjie/UniStock/dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='UTSD-full-npy', help='data file')
    parser.add_argument('--scale', type=bool, default=True, help='data scale')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='close', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--train_begin_date', type=str, default='2013-01-01', help='train dataset begin date')
    parser.add_argument('--valid_begin_date', type=str, default='2016-01-01', help='valid dataset begin date')
    parser.add_argument('--test_begin_date', type=str, default='2017-01-01', help='test dataset begin date')
    parser.add_argument('--test_end_date', type=str, default='2018-01-01', help='test dataset begin date')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')
    
    # autoregressive configs
    parser.add_argument('--token_num', type=int, default=56, help='input token num')
    parser.add_argument('--input_token_len', type=int, default=12, help='input token length')
    parser.add_argument('--output_token_len', type=int, default=12, help='input token length')
    parser.add_argument('--autoregressive', action='store_true', help='autoregressive', default=False)
    parser.add_argument('--output_len', type=int, default=10, help='output len')
    
    args = parser.parse_args()

    args.seq_len = args.token_num * args.input_token_len
    print(f'args: {args}')

    flag = 'train'
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag in ['test', 'val']:
        shuffle_flag = False
    else:
        shuffle_flag = True
    
    data_set = UTSD_Npy(
            args=args,
            market_name=args.market_name,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            input_token_len=args.input_token_len,
            output_token_len=args.output_token_len,
            autoregressive=args.autoregressive,
            size=[args.seq_len, args.label_len, args.output_len] if flag == 'test' else [args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=args.freq
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=False)

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(data_loader)):
        print(f'i {i} batch_x {batch_x.shape} batch_y {batch_y.shape} batch_x_mark {batch_x_mark.shape} batch_y_mark {batch_y_mark.shape}')