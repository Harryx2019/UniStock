import os
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

'''
    数据集包含五个文件夹
    - price_data 一个股票市场一个csv,包含每日各维度原始价格数据
    - features 一个股票市场一个csv,包含每日各维度特征数据
    - scaler 一个股票市场一个csv, 包含标准化的数据【例如NASDAQ&NYSE 为训练集最大值】
    - stock_list 一个股票市场一个csv, 包含所有股票名 【todo: 参照qlib, 每支股票有效交易日期】
    - trade_dates 一个股票市场一个csv, 包含有效交易日期, 数据需要保证同一股票市场所有交易日期都有数据, 实盘无数据填充为NaN
'''

class Dataset_Stock(Dataset):
    def __init__(self, args, market_name, root_path, data_path='features', flag='train', 
                 input_token_len=96, output_token_len=96, autoregressive=False,
                 size=None, features='MS', stride=1, 
                 target='close', scale=False, timeenc=0, freq='d'):
        '''
            功能: 读取指定市场的全部股票数据, 构建模型需要的样本: 时序特征、时间戳特征、样本掩码、真实标签

            args:           配置json
            market_name:    股票市场名称/数据集名称
            root_path:      数据集根目录
            data_path:      数据(特征)目录
            flag:           数据集划分
            input_token_len:   输入token长度
            output_token_len:  输出token长度
            autoregressive:    是否构建自回归数据集
            size:           [seq_len, label_len, pred_len]
            features:       模式[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
            target:         目标列
            scale:          是否标准化,根据训练集的数据进行标准化【NASDAQ&NYSE 在构建数据集的时候已经根据训练集的最大值进行标准化了 TODO:将该标准化在内部实现】
            timeenc:        时间特征编码方法 1:time_features编码 0:时间戳分解
            freq:           数据频率
            stride:         数据采样步长
        '''
        self.args = args
        # info
        if size == None:
            raise ValueError(f'Dataset_Stock size is None!!')
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # 24-11-14 自回归训练数据集
        self.input_token_len = input_token_len
        self.output_token_len = output_token_len
        self.autoregressive = autoregressive

        # 24-11-29 数据集时间划分
        self.train_begin_date = self.args.train_begin_date
        self.valid_begin_date = self.args.valid_begin_date
        self.test_begin_date = self.args.test_begin_date
        self.test_end_date = self.args.test_end_date

        # 24-12-20 收益率mask比率
        self.quantile = self.args.quantile

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        print(f'set_type: {self.set_type}')

        self.features = features
        self.target = target
        self.scale = scale
        self.scalers = {}  # 存储每只股票的 scaler
        self.timeenc = timeenc
        self.freq = freq
        self.stride = stride

        self.market_name = market_name
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def __read_data__(self):
        # 读取股票列表
        stock_list = pd.read_csv(os.path.join(self.root_path, 'stock_list', f'{self.market_name}.csv'), header=None, names=['stock'])
        self.stock_list = stock_list['stock'].values.tolist()
        self.stock_list.sort()
        print('#stocks selected:', len(self.stock_list))

        # 读取模型输入数据 [原始价格数据 或 特征数据]
        df_raw_path = os.path.join(self.root_path, self.data_path, f'{self.market_name}.csv')
        df_raw = pd.read_csv(df_raw_path)
        '''
            df_raw.columns: ['stock', 'date', ...(other features), target feature]
        '''
        # 25-1-4 为了加快训练和测试速度, 股票数据只取close列
        cols = []
        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('stock')
        # cols.remove('date')
        print(f'#successfully read df_raw from {df_raw_path} data shape {df_raw.shape}')
        print(f'#cols: {cols}')
        df_raw = df_raw[['stock', 'date'] + cols + [self.target]]
        
        # 对数据进行排序(有点耗时)
        # time_now = time.time()
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        # df_raw = df_raw.sort_values(by=['stock', 'date'])
        # df_raw = df_raw.reset_index(drop=True)
        # print(f'#sort df_raw cost: {(time.time() - time_now):.4f}s')

        stock_list_ = df_raw['stock'].unique().tolist()
        stock_list_.sort()
        if self.stock_list != stock_list_:
            raise ValueError(f"Stock lists are not the same between [stock_list] and [{self.data_path}]")

        # 以第一只股票的数据作为样本统计数据长度
        df_sample = df_raw[df_raw['stock'] == self.stock_list[0]]
        df_sample = df_sample[df_sample['date'] >= self.train_begin_date]
        df_sample = df_sample[df_sample['date'] < self.test_end_date]
        df_sample = df_sample.reset_index(drop = True)
        
        num_train = len(df_sample[df_sample['date'] < self.valid_begin_date])
        num_test = len(df_sample[df_sample['date'] >= self.test_begin_date])
        num_vali = len(df_sample) - num_train - num_test
        print(f'num_train {num_train} num_vali {num_vali} num_test {num_test}')
        border1s = [0, num_train - self.seq_len, len(df_sample) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_sample)]
        print(f'border1s {border1s} border2s {border2s}')
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 以第一只股票的数据作为样本计算时间特征
        df_stamp = df_sample[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            self.data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            self.data_stamp = data_stamp.transpose(1, 0)

        # 计算 mask 和 return
        def calculate_mask_return(df):
            df = df.copy()
            df['mask'] = df['close'].notna().astype(int)  # mask: 1 表示非 NaN，0 表示 NaN 24-11-9: mask仅针对close数据, 用于自监督学习中
            for j in range(self.pred_len):
                df[f'return_{j + 1}'] = df['close'].pct_change(periods = j + 1, fill_method=None)  # 计算收益率
                df[f'return_{j + 1}'] = df[f'return_{j + 1}'].shift(-1 * j) # shift -1*j 是便于ground_truth的构建

                # 2024-12-04 计算前后2.5%的分位数 剔除涨跌幅较大的股票
                lower_percentile = df[f'return_{j + 1}'].quantile(self.quantile)
                upper_percentile = df[f'return_{j + 1}'].quantile(1 - self.quantile)
                df[f'return_{j + 1}'] = df[f'return_{j + 1}'].apply(lambda x: np.nan if x < lower_percentile or x > upper_percentile else x)
            return df
        
        return_cols = [f'return_{j + 1}' for j in range(self.pred_len)]
        print(f'#return cols: {return_cols}')
        print(f'#drop return out of quantile {self.quantile}')
        
        # 初始化容器
        eod_data, masks, ground_truth, base_price = [], [], [], []

        time_now = time.time()
        for stock, group in df_raw.groupby('stock'):
            group = group[group['date'] >= self.train_begin_date]
            group = group[group['date'] < self.test_end_date]
            # 将[特征列, 标签列]提取出来
            df_data = group[cols + [self.target]]
            if self.scale:
                # 24-11-26 由于对输入特征进行StandardScaler标准化之后, 标准化前后计算的return数值会发生变化, 因此更改为训练集最大值初始化
                df_train = df_data[border1s[0]:border2s[0]]
                max_value = df_train.max() 
                # 24/12-21 若某只股票df_train全为nan, max_value也为nan, 数据除nan值会导致数据全为nan, 因此改为用第一个值进行标准化
                if max_value.isna().sum() > 0:
                    max_value = df_data.apply(lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan)
                
                # if max_value.eq(0).sum() > 0:
                #     print(f'stock {stock} max_value has 0')
                # elif max_value.isna().sum() > 0:
                #     print(f'stock {stock} max_value has nan!')
                
                df_data = df_data / max_value
                self.scalers[stock] = max_value
                
                # 标准化数据
                # scaler = StandardScaler()
                # df_train = df_data[border1s[0]:border2s[0]]
                # df_data = df_data.fillna(df_train.mean()) # 为使得缺失值数据标准化处理后为0, 填充为均值
                # scaler.fit(df_train.values)
                # self.scalers[stock] = scaler
                # data = scaler.transform(df_data.values)
            else:
                raise NotImplementedError
            
            # 计算 mask 和 return
            df_data_r = calculate_mask_return(df_data)
            df_data_r = df_data_r[border1:border2]
            masks.append(df_data_r['mask'].values)
            ground_truth.append(df_data_r[return_cols].values)
            base_price.append(df_data_r['close'].values)

            df_data = df_data[border1:border2]
            # 重要！ 缺失数据如何填充
            df_data = df_data.fillna(0.0)
            eod_data.append(df_data.values)
        print(f'#prepare dataset cost: {(time.time() - time_now):.4f}s')
        
        self.eod_data = np.array(eod_data)
        self.masks = np.array(masks)
        self.ground_truth = np.array(ground_truth)
        self.base_price = np.array(base_price)

        print('#eod_data shape:', self.eod_data.shape)          # [num_stock, dataset_len, n_vars]
        print('#masks shape:', self.masks.shape)                # [num_stock, dataset_len]
        print('#ground_truth shape:', self.ground_truth.shape)  # [num_stock, dataset_len, pred_len]
        print('#base_price shape:', self.base_price.shape)      # [num_stock, dataset_len]
        print('#data_stamp shape:', self.data_stamp.shape)      # [dataset_len, time_features]
       
        self.batch_size = self.eod_data.shape[0]
        # 25-1-4 数据采样
        self.n_timepoint = (self.eod_data.shape[1] - self.seq_len - self.pred_len) // self.stride + 1
        self.n_var = self.eod_data.shape[2]
        print(f'batch_size: {self.batch_size} n_var: {self.n_var} n_timepoint: {self.n_timepoint}')
    
    def __getitem__(self, index):
        '''
            截面数据组织为batch, 超参数batch_size需要设置为1
        '''
        assert self.args.batch_size == 1
        
        s_begin = index * self.stride # TODO: 数据是否会越界？
        s_end = s_begin + self.seq_len

        if self.autoregressive:
            '''
                next-token prediction
                自回归Decoder-only预训练输入输出
                input_token_len 与 output_token_len 可不相同
                TODO: TimesFM的实现方法
            '''
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            # 输入特征
            seq_x = self.eod_data[:, s_begin:s_end, :]  # Encoder输入
            # 滑动窗口采样, input_token_len != output_token_len
            seq_y = self.eod_data[:, r_begin:r_end, :]                                      # [num_stocks, seq_len - input_token_len + output_token_len, n_vars]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=1, size=self.output_token_len,                   # [num_stocks, num_windows, n_vars, output_token_len]
                                 step=self.input_token_len).permute(0, 1, 3, 2)             # [num_stocks, num_windows, output_token_len, n_vars]
            seq_y = seq_y.reshape(self.batch_size, seq_y.shape[1] * seq_y.shape[2], -1)     # [num_stocks, num_windows * output_token_len, n_vars]

            # 时间协特征
            seq_x_mark = self.data_stamp[s_begin:s_end, :]   # Encoder输入
            seq_y_mark = self.data_stamp[r_begin:r_end, :]   # Decoder输入
            seq_y_mark = torch.tensor(seq_y_mark)
            seq_y_mark = seq_y_mark.unfold(dimension=0, size=self.output_token_len,            # [num_windows, time_vars, output_token_len]
                                            step=self.input_token_len).permute(0, 2, 1)        # [num_windows, output_token_len, time_vars]
            seq_y_mark = seq_y_mark.reshape(seq_y_mark.shape[0] * seq_y_mark.shape[1], -1)     # [num_windows * output_token_len, time_vars]

            # mask close=nan时mask=0 自监督学习中使用
            seq_x_mask = self.masks[:, s_begin:s_end]
            seq_y_mask = self.masks[:, r_begin:r_end]                                                           # [num_stocks, seq_len - input_token_len + output_token_len]
            seq_y_mask = torch.tensor(seq_y_mask)
            seq_y_mask = seq_y_mask.unfold(dimension=1, size=self.output_token_len,                             # [num_stocks, num_windows, output_token_len]
                                            step=self.input_token_len)
            seq_y_mask = seq_y_mask.reshape(self.batch_size, seq_y_mask.shape[1] * seq_y_mask.shape[2])         # [num_stocks, num_windows * output_token_len]
        
        else:
            '''
                标准Encoder-Decoder输入输出
            '''
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            # 输入特征
            seq_x = self.eod_data[:, s_begin:s_end, :]  # Encoder输入
            seq_y = self.eod_data[:, r_begin:r_end, :]  # Decoder输入

            # 时间协特征
            seq_x_mark = self.data_stamp[s_begin:s_end, :]   # Encoder输入
            seq_y_mark = self.data_stamp[r_begin:r_end, :]   # Decoder输入

            # mask close=nan时mask=0 预训练自监督学习中使用
            seq_x_mask = self.masks[:, s_begin:s_end]
            seq_y_mask = self.masks[:, r_begin:r_end]
        
        # 24/12-23 历史数据缺失值占模型输入长度一半的股票将收盘价直接设置为nan
        threshold = self.seq_len // 2
        zero_count = np.sum(seq_x[:,:,-1] == 0, axis=1) # [num_stocks]
        stocks_above_threshold = np.where(zero_count > threshold)[0]
        
        # 当天的收盘价, 用于计算预测收益率
        price_batch = np.expand_dims(self.base_price[:, s_end - 1], axis=1)
        price_batch[stocks_above_threshold] = np.nan

        # 从当天开始的周期为pred_len的收益率序列
        # 用于1. 模型回测  2. 基于股票排序损失的下游任务微调
        gt_batch = self.ground_truth[:, s_end, :]
        
        '''
            1. 特征数据
            seq_x shape: [num_stocks, seq_len, feature_dim]
            seq_y shape: [num_stocks, (label_len + )pred_len, feature_dim] (label_len = seq_len - pred_len)

            2. 时间特征(协变量)数据
            seq_x_mark shape: [num_stocks, seq_len, time_features]
            seq_y_mark shape: [num_stocks, (label_len + )pred_len, time_features]

            3. mask数据
            seq_x_mask shape: [num_stocks, seq_len]
            seq_y_mask shape: [nums_stocks, (label_len + )pred_len]

            4. 收盘价序列(用于计算预测收益率)【包含nan值】
            price_batch shape: [num_stocks, 1]

            5. 真实收益率序列【包含nan值】
            gt_batch shape: [num_stocks, pred_len]
        '''
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_mask, seq_y_mask, price_batch, gt_batch

    def __len__(self):
        return self.n_timepoint
    
    def inverse_transform(self, data):
        assert data.shape[0] == len(self.stock_list)
        for i in range(len(self.stock_list)):
            data[i] = self.scalers[self.stock_list[i]] * data[i]
        return data
    

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test data_stock')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    # data loader
    parser.add_argument('--data', type=str, default='Stock', help='dataset type')
    parser.add_argument('--market_name', type=str, default='A_share', help='stock market name')
    parser.add_argument('--root_path', type=str, default='/home/xiahongjie/UniStock/dataset/stock', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='price_data', help='data file')
    parser.add_argument('--scale', type=bool, default=True, help='data scale')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='close', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--train_begin_date', type=str, default='2016-01-01', help='train dataset begin date')
    parser.add_argument('--valid_begin_date', type=str, default='2019-01-01', help='valid dataset begin date')
    parser.add_argument('--test_begin_date', type=str, default='2020-01-01', help='test dataset begin date')
    parser.add_argument('--test_end_date', type=str, default='2021-01-01', help='test dataset begin date')
    parser.add_argument('--quantile', type=float, default=0.01, help='mask ratio of return ratio')
    parser.add_argument('--stride_dataset', type=int, default=1, help='few shot')

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

    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = True

    data_set = Dataset_Stock(
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
            freq=args.freq,
            stride=args.stride_dataset
        )
    
    print(f'flag={flag}, n_vars={data_set.n_var}, time_points={data_set.n_timepoint}, len(data_set)={len(data_set)}')

    data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=False)
    
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, 
            batch_x_mask, batch_y_mask, price_batch, gt_batch) in enumerate(tqdm(data_loader)):
        # print(f'i:{i} batch_x:{batch_x.shape} batch_y:{batch_y.shape} '\
        #         f'batch_x_mark:{batch_x_mark.shape} batch_y_mark:{batch_y_mark.shape} '\
        #         f'batch_x_mask:{batch_x_mask.shape} batch_y_mask:{batch_y_mask.shape} '\
        #         f'price_batch:{price_batch.shape} gt_batch:{gt_batch.shape}')
        pass
        # break