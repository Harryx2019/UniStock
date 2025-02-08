import argparse
import os
import torch
from exp.exp_stock import Exp_Stock
from utils.print_args import print_args
import random
import numpy as np
import torch.distributed as dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniStock')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--pretrain_data', type=str, default='UTSD', help='pretrain data')
    
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--market_name', type=str, default='NASDAQ', help='stock market name')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--scale', action='store_true', default=False, help='data scale')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--train_begin_date', type=str, default='2013-01-01', help='train dataset begin date')
    parser.add_argument('--valid_begin_date', type=str, default='2016-01-01', help='valid dataset begin date')
    parser.add_argument('--test_begin_date', type=str, default='2017-01-01', help='test dataset begin date')
    parser.add_argument('--test_end_date', type=str, default='2018-01-01', help='test dataset end date')
    parser.add_argument('--quantile', type=float, default=0.01, help='mask ratio of return ratio')
    parser.add_argument('--stride_dataset', type=int, default=1, help='few shot')
    
    # forecasting task
    parser.add_argument('--token_num', type=int, default=7, help='input token num')
    parser.add_argument('--input_token_len', type=int, default=96, help='input token length')
    parser.add_argument('--output_token_len', type=int, default=96, help='input token length')
    parser.add_argument('--autoregressive', action='store_true', help='autoregressive', default=False)
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--output_len', type=int, default=5, help='prediction sequence length')
    parser.add_argument('--patch_len', type=int, default=12, help='the length of patch')
    parser.add_argument('--stride', type=int, default=12, help='the length of stride')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48, help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--graph',  action='store_true', default=False, help='use graph')
    parser.add_argument('--num_hyperedges', type=int, default=100, help='the number of hyperedge of DyHGAT')
    parser.add_argument('--hyperedges_quantile', type=float, default=0.9, help='the quantile of hyperedge of DyHGAT')


    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--alpha', type=int, default=1, help='alpha of mse_rank_loss')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # VisionTS
    parser.add_argument('--vm_pretrained', type=int, default=1)
    parser.add_argument('--vm_ckpt', type=str, default="./ckpt/")
    parser.add_argument('--vm_arch', type=str, default='mae_base')
    parser.add_argument('--ft_type', type=str, default='ln')
    parser.add_argument('--periodicity', type=int, default=0)
    parser.add_argument('--interpolation', type=str, default='bilinear')
    parser.add_argument('--norm_const', type=float, default=0.4)
    parser.add_argument('--align_const', type=float, default=0.4)
    parser.add_argument('--save_dir', type=str, default='.', help='save dir')

    args = parser.parse_args()
    
    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False
    print(f'torch.cuda.is_available():{torch.cuda.is_available()}')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print_args(args)

    if args.task_name in ['pretrain', 'finetune', 'zeroshot', 'supervised']:
        Exp = Exp_Stock
    else:
        raise NotImplementedError

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)

            setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_graph{}_a{}_q{}_ne{}_hq{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.market_name,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.graph,
                args.alpha,
                args.quantile,
                args.num_hyperedges,
                args.hyperedges_quantile,
                args.des, ii)

            if args.task_name == 'pretrain':
                print('>>>>>>>start pre-training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                if args.data == 'Stock': # dataloader
                    # 25-1-5 有时间再做
                    # exp.pretrain_stock(setting)
                    raise NotImplementedError
                elif args.data == 'Utsd_Npy':
                    if args.model == 'Timer':
                        if args.market_name in ['Stock', 'A_share', 'S&P500', 'Stock_maxscale', 'A_share_maxscale', 'S&P500_maxscale']:
                            # 25-1-5 对Timer进行持续预训练
                            # weight_path = os.path.join('./checkpoints', 'Timer', 'Timer_forecast_1.0.pth')
                            weight_path = f'/home/xiahongjie/UniStock/checkpoints/Timer/{args.pretrain_data}'\
                                        f'/UTSD-full-npy/pretrain_{args.pretrain_data}_sl672_it{args.input_token_len}_Timer_Utsd_Npy_'\
                                        f'{args.pretrain_data}_ftM_sl672_ll{args.label_len}_pl5_dm1024_nh8_el8_dl1_df2048_graphFalse_a1_q0.01_ne100_hq0.9_Exp_0/checkpoint.pth'
                        elif args.market_name in ['UTSD', 'Stock_only', 'A_share_only', 'S&P500_only', 
                                                  'Stock_only_maxscale', 'A_share_only_maxscale', 'S&P500_only_maxscale']:
                            # 重头开始预训练
                            weight_path = None
                        else:
                            raise NotImplementedError
                    elif args.model == 'CI_STHPAN':
                        if args.market_name in ['Stock', 'A_share', 'S&P500', 'Stock_maxscale', 'A_share_maxscale', 'S&P500_maxscale']:
                            # 25-1-5 对UTSD预训练后的模型进行持续预训练
                            # weight_path = '/home/xiahongjie/UniStock/checkpoints/CI_STHPAN/UTSD/UTSD-full-npy/pretrain_UTSD_sl672_it96_CI_STHPAN_Utsd_Npy_UTSD_ftM_sl672_ll0_pl5_dm1024_nh8_el8_dl1_df2048_graphFalse_Exp_0/checkpoint.pth'
                            weight_path = f'/home/xiahongjie/UniStock/checkpoints/CI_STHPAN/{args.pretrain_data}'\
                                        f'/UTSD-full-npy/pretrain_{args.pretrain_data}_sl672_it{args.input_token_len}_CI_STHPAN_Utsd_Npy_'\
                                        f'{args.pretrain_data}_ftM_sl672_ll{args.label_len}_pl5_dm1024_nh8_el8_dl1_df2048_graphFalse_a1_q0.01_ne100_hq0.9_Exp_0/checkpoint.pth'
                        elif args.market_name in ['UTSD', 'Stock_only', 'A_share_only', 'S&P500_only', 
                                                  'Stock_only_maxscale', 'A_share_only_maxscale', 'S&P500_only_maxscale']:
                            # 重头开始预训练
                            weight_path = None
                        else:
                            raise NotImplementedError
                    else:
                        weight_path = None
                    exp.pretrain_ts(setting, weight_path)
                else:
                    raise NotImplementedError
            elif args.task_name == 'finetune':
                print('>>>>>>>start fine-tuning : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.finetune(setting)
                
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # for type in ['linear_probe', 'entire']:
                for type in ['linear_probe']:
                    # 微调方法有两类模型参数: entire / linear_probe
                    print(f'type: {type}\n')
                    weight_path = os.path.join(args.checkpoints, setting, args.test_begin_date, type, 'checkpoint.pth')
                    setting_ = setting + '_ty' + type
                    exp.test(setting_, weight_path=weight_path)
            elif args.task_name == 'supervised':
                print('>>>>>>>start supervised : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.supervised(setting)
                
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                weight_path = os.path.join(args.checkpoints, setting, args.test_begin_date, 'checkpoint.pth')
                exp.test(setting, weight_path=weight_path)
            else:
                raise NotImplementedError

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_graph{}_a{}_q{}_ne{}_hq{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.market_name,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.graph,
            args.alpha,
            args.quantile,
            args.num_hyperedges,
            args.hyperedges_quantile,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        if args.task_name == 'zeroshot':
            # zeroshot 测试 1. 官方预训练模型  2. 自定义预训练模型
            if args.model == 'Timer':
                if args.pretrain_data in ['UTSD', 'Stock', 'A_share', 'S&P500',
                                             'Stock_maxscale', 'A_share_maxscale', 'S&P500_maxscale',
                                             'Stock_only', 'A_share_only', 'S&P500_only',
                                             'Stock_only_maxscale', 'A_share_only_maxscale', 'S&P500_only_maxscale']:
                    weight_path = f'./checkpoints/Timer/{args.pretrain_data}'\
                                f'/UTSD-full-npy/pretrain_{args.pretrain_data}_sl672_it{args.input_token_len}_Timer_Utsd_Npy_'\
                                f'{args.pretrain_data}_ftM_sl672_ll{args.label_len}_pl5_dm1024_nh8_el8_dl1_df2048_graphFalse_a1_q0.01_ne100_hq0.9_Exp_0/checkpoint.pth'
                # elif args.pretrain_data == 'UTSD':
                #     weight_path = os.path.join('./checkpoints', 'Timer', 'Timer_forecast_1.0.pth')
                else:
                    raise NotImplementedError
            elif args.model == 'VisionTS':
                weight_path = None # 自动加载
            elif args.model == 'TimesFM':
                weight_path = os.path.join('./checkpoints', 'TimesFM', 'timesfm-1.0-200m-pytorch.ckpt')
            elif args.model == 'Moirai':
                weight_path = None # 自动加载
            elif args.model == 'TimeMOE':
                weight_path = None # 自动加载
            else:
                raise NotImplementedError
            exp.test(setting, weight_path=weight_path)
        elif args.task_name == 'finetune':
            # 微调方法有两类模型参数: entire / linear_probe
            for type in ['linear_probe', 'entire']:
                print(f'type: {type}\n')
                weight_path = os.path.join(args.checkpoints, setting, args.test_begin_date, type, 'checkpoint.pth')
                setting_ = setting + '_ty' + type
                exp.test(setting_, weight_path=weight_path)
        elif args.task_name == 'supervised':
            weight_path = os.path.join(args.checkpoints, setting, args.test_begin_date, 'checkpoint.pth')
            exp.test(setting, weight_path=weight_path)
        else:
            raise NotImplementedError
        
        torch.cuda.empty_cache()
