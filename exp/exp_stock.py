from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import backtest, mse_rank_loss, reconstruction_mse_loss, autoregression_mse_loss
from torch.nn.parallel import DistributedDataParallel as DDP
from models.TimesFM import TimesFMConfig
from models.momentfm import MOMENTPipeline

import os
import time
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np

class Exp_Stock(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stock, self).__init__(args)

    def _build_model(self):
        if self.args.model == 'TimesFM':
            # 25-1-14 TimesFM的参数按照官方定义的配置
            model_config = TimesFMConfig()
            model = self.model_dict[self.args.model].Model(model_config).float()
        elif self.args.model == 'MOMENT':
            model = MOMENTPipeline.from_pretrained(
                "AutonLab/MOMENT-1-large", 
                model_kwargs={
                    'task_name': 'forecasting',
                    'forecast_horizon': self.args.pred_len,
                    'head_dropout': 0.1,
                    'weight_decay': 0,
                    'freeze_encoder': True, # Freeze the patch embedding layer
                    'freeze_embedder': True, # Freeze the transformer encoder
                    'freeze_head': False, # The linear forecasting head must be trained
                },
                # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
            )
            model.init()
        elif self.args.model == 'Moirai':
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-large"),
                prediction_length=self.args.pred_len,
                context_length=self.args.seq_len,
                patch_size=32,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()
        print('number of model params', sum(p.numel() for p in model.parameters()))
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # for name, param in model.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()}")
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        if self.args.task_name == 'pretrain':
            if self.args.model == 'CI_STHPAN':
                criterion = reconstruction_mse_loss
            elif self.args.model == 'Timer':
                criterion = autoregression_mse_loss
            else:
                raise NotImplementedError
        elif self.args.task_name == 'finetune' or self.args.task_name == 'supervised':
            criterion = mse_rank_loss
        else:
            raise NotImplementedError
        return criterion
    
    def vali_ts(self, vali_data, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # print(f'i {i} batch_x {batch_x.shape} batch_y {batch_y.shape} batch_x_mark {batch_x_mark.shape} batch_y_mark {batch_y_mark.shape}')
                batch_x = batch_x.float().to(self.device)
                # batch_y = batch_y.float()
                # batch_x_mark = batch_x_mark.float()
                # batch_y_mark = batch_y_mark.float()
                
                if self.args.model == 'CI_STHPAN':
                    '''
                        CI_STHPAN基于encoder对数据进行Mask Reconstruction预训练,
                        对输入的batch_x [batch_size, seq_len(token_num * input_token_len), n_vars] 进行token级别随机mask后进行重构
                    '''
                    x, dec_out, x_mask = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    # x       [bs x token_num x n_vars x input_token_len]   输入token
                    # dec_out [bs x token_num x n_vars x input_token_len]   预测token
                    # mask    [bs x token_num x n_vars]                     token mask
                    if i == 0:
                        print(f' x {x.shape}\ndec_out {dec_out.shape}\nx_mask {x_mask.shape}\n')
                    loss = criterion(x, dec_out, x_mask) # reconstruction_mse_loss
                elif self.args.model == 'Timer':
                    '''
                        Timer基于decoder对数据进行Auto Regression预训练
                        对输入的batch_x [batch_size, seq_len(token_num * input_token_len), n_vars] 进行token级别casual预测
                    '''
                    batch_y = batch_y.float().to(self.device)
                    outputs, attns = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    # outputs [batch_size, pred_len(token_num * output_token_len), n_vars]
                    if i == 0:
                        print(f'outputs {outputs.shape}\nattns {attns[-1].shape}\n')
                    loss = criterion(outputs, batch_y) # autoregression_mse_loss
                else:
                    raise NotImplementedError
                
                total_loss.append(loss.item())
        
        # 24-12/19 不返回损失平均值
        # total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def pretrain_ts(self, setting, weight_path=None):
        if weight_path != None:
            self.transfer_weights(weight_path)
        else:
            print(f'pretrain model from scratch')

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_loss_epoch = []
        val_loss_epoch = []
        test_loss_epoch = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # print(f'i {i} batch_x {batch_x.shape} batch_y {batch_y.shape} batch_x_mark {batch_x_mark.shape} batch_y_mark {batch_y_mark.shape}')
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                # batch_y = batch_y.float() # encoder reconstruction 不需要, decoder autoregression 需要
                # batch_x_mark = batch_x_mark.float()
                # batch_y_mark = batch_y_mark.float()
                
                if self.args.model == 'CI_STHPAN': 
                    '''
                        CI_STHPAN基于encoder对数据进行Mask Reconstruction预训练,
                        对输入的batch_x [batch_size, seq_len(token_num * input_token_len), n_vars] 进行token级别随机mask后进行重构
                    '''
                    x, dec_out, x_mask = self.model(batch_x, None, None, None)
                    # x       [bs x token_num x n_vars x input_token_len]   输入token
                    # dec_out [bs x token_num x n_vars x input_token_len]   预测token
                    # mask    [bs x token_num x n_vars]                     token mask
                    if epoch == 0 and i == 0:
                        print(f'x {x.shape}\ndec_out {dec_out.shape}\nx_mask {x_mask.shape}\n')
                    loss = criterion(x, dec_out, x_mask) # reconstruction_mse_loss
                
                elif self.args.model == 'Timer':
                    '''
                        Timer基于decoder对数据进行Auto Regression预训练
                        对输入的batch_x [batch_size, seq_len(token_num * input_token_len), n_vars] 进行token级别casual预测
                    '''
                    batch_y = batch_y.float().to(self.device)
                    outputs, attns = self.model(batch_x, None, None, None)
                    if epoch == 0 and i == 0:
                        print(f'batch_x {batch_x.shape}\noutputs {outputs.shape}\nattns[-1] {attns[-1].shape}\n')
                    # outputs [batch_size, pred_len(token_num * output_token_len), n_vars]
                    loss = criterion(outputs, batch_y) # autoregression_mse_loss

                else:
                    raise NotImplementedError
                
                train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            vali_loss = self.vali_ts(vali_data, vali_loader, criterion)
            test_loss = self.vali_ts(test_data, test_loader, criterion)
            
            train_loss_epoch.append(train_loss)
            val_loss_epoch.append(vali_loss)
            test_loss_epoch.append(test_loss)

            train_loss = np.average(train_loss)
            vali_loss = np.average(vali_loss)
            test_loss = np.average(test_loss)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        train_loss_epoch = np.array(train_loss_epoch)
        val_loss_epoch = np.array(val_loss_epoch)
        test_loss_epoch = np.array(test_loss_epoch)
        print(f'# train_loss_epoch {train_loss_epoch.shape} val_loss_epoch {val_loss_epoch.shape} test_loss_epoch {test_loss_epoch.shape}')
        
        np.save(path + '/' + 'train_loss_epoch.npy', train_loss_epoch)
        np.save(path + '/' + 'val_loss_epoch.npy', val_loss_epoch)
        np.save(path + '/' + 'test_loss_epoch.npy', test_loss_epoch)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.load_model(best_model_path)

        return self.model


    def vali_stock(self, vali_data, vali_loader, criterion):
        total_loss = []
        if self.args.task_name == 'finetune' or self.args.task_name == 'supervised':
            total_reg_loss = []
            total_rank_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, 
                    batch_x_mask, batch_y_mask, price_batch, gt_batch) in enumerate(tqdm(vali_loader)):
                # print(f'i:{i} batch_x:{batch_x.shape} batch_y:{batch_y.shape} '\
                #     f'batch_x_mark:{batch_x_mark.shape} batch_y_mark:{batch_y_mark.shape} '\
                #     f'batch_x_mask:{batch_x_mask.shape} batch_y_mask:{batch_y_mask.shape} '\
                #     f'price_batch:{price_batch.shape} gt_batch:{gt_batch.shape}')
                batch_x = batch_x[0].float()
                price_batch = price_batch[0].float()
                gt_batch = gt_batch[0].float()

                # 以下数据未使用
                # batch_y = batch_y[0].float()#.to(self.device)
                # batch_x_mark = batch_x_mark[0].float()#.to(self.device)
                # batch_y_mark = batch_y_mark[0].float()#.to(self.device)
                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()#.to(self.device)

                # 过滤无效数据
                mask = ~torch.isnan(price_batch.squeeze()) # [num_stock] 当日有效交易股票数
                batch_x = batch_x[mask]
                price_batch = price_batch[mask].to(self.device)
                gt_batch = gt_batch[mask].to(self.device)

                num_stocks = batch_x.shape[0]
                # print(f'i:{i} valid num_stock:{num_stocks}')

                # encoder - decoder
                if self.args.task_name == 'pretrain':
                    if self.args.model == 'CI_STHPAN':
                        # x 输入patch  dec_out 预测patch  x_mask patch mask
                        x, dec_out, x_mask = self.model(batch_x, None, None, None)
                        loss = criterion(x, dec_out, x_mask) # reconstruction_mse_loss
                        total_loss.append(loss.item())
                    else:
                        raise NotImplementedError
                elif self.args.task_name == 'finetune' or self.args.task_name == 'supervised':
                    split_steps = num_stocks // self.split_batch
                    if num_stocks % self.split_batch != 0:
                        split_steps += 1
                    outputs_k = []
                    # k循环输入样本数
                    for k in range(split_steps):
                        batch_x_k = batch_x[k * self.split_batch : (k + 1) * self.split_batch, :, :].to(self.device)
                        if self.args.model == 'VisionTS':
                            outputs = self.model(batch_x_k, None, None, None)
                        elif self.args.model == 'Transformer':
                            outputs = self.model(batch_x_k) 
                        elif self.args.model == 'LSTM':
                            outputs = self.model(batch_x_k) 
                        elif self.args.model == 'Timer':
                            outputs, attns = self.model(batch_x_k, None, None, None)
                            # 25-1/2 Timer的输出为token_num * token_len, 应该用预测的最后一个token的前pred_len个元素作为模型输出结果
                            outputs = outputs[:, (self.args.token_num - 1) * self.args.output_token_len:, :]
                        elif self.args.model == 'CI_STHPAN':
                            # x_enc, x_mark_enc, x_dec, x_mark_dec
                            outputs, attns = self.model(batch_x_k, None, None, None)
                            outputs = outputs[:, (self.args.token_num - 1) * self.args.output_token_len:, :]
                        elif self.args.model == 'MOMENT':
                            batch_x_k = batch_x_k.permute(0, 2, 1) # [B, C, L]
                            outputs = self.model(x_enc=batch_x_k)
                            outputs = outputs.forecast # [B, C, T]
                            outputs = outputs.permute(0, 2, 1) # [B, T, C]
                        elif self.args.model == 'Master':
                            outputs = self.model(x=batch_x_k)
                        elif self.args.model == 'DTML':
                            outputs = self.model(stocks=batch_x_k)
                        else:
                            raise NotImplementedError

                        if i == 0:
                            print(f'k {k} batch_x {batch_x.shape} split_batch {self.split_batch} batch_x_k {batch_x_k.shape} outputs {outputs.shape}')
                        outputs_k.append(outputs)
                        # torch.cuda.empty_cache()
                    outputs = torch.cat(outputs_k, dim=0)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    # 24-11-10 注意test_loader pred_len和output_len的含义不同
                    # 25-1/2 取前pred_len个预测值 [这里是和time series libary目的不同的地方]
                    # 25-1/5 对于encoder-only, 模型直接预测pred_len; 而对于decoder-only, 模型预测下一个token
                    outputs = outputs[:, :self.args.pred_len, f_dim:]

                    if i == 0:
                        print(f' batch_x {batch_x.shape}\noutputs {outputs.shape}\n')

                    # mse_rank_loss
                    loss, reg_loss, rank_loss, rr = criterion(i, outputs, price_batch, gt_batch,
                                                              self.args.alpha, self.device)

                    total_reg_loss.append(reg_loss.item())
                    total_rank_loss.append(rank_loss.item())
                    total_loss.append(loss.item())
                else:
                    raise NotImplementedError
        
        # 24-12/25 为便于损失的可视, 不对loss取均值
        # total_loss = np.average(total_loss)
        # if self.args.task_name == 'finetune':
        #     total_reg_loss = np.average(total_reg_loss)
        #     total_rank_loss = np.average(total_rank_loss)

        self.model.train()
        if self.args.task_name == 'pretrain':
            return total_loss
        elif self.args.task_name == 'finetune' or self.args.task_name == 'supervised':
            return total_loss, total_reg_loss, total_rank_loss

    def pretrain_stock(self, setting):
        '''
            仅对股票数据按截面进行pretrain
            TODO: 时间不一定来得及做实验
        '''
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_loss_epoch = []
        val_loss_epoch = []
        test_loss_epoch = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, 
                    batch_x_mask, batch_y_mask, price_batch, gt_batch) in enumerate(train_loader):
                # print(f'i:{i} batch_x:{batch_x.shape} batch_y:{batch_y.shape} '\
                #     f'batch_x_mark:{batch_x_mark.shape} batch_y_mark:{batch_y_mark.shape} '\
                #     f'batch_x_mask:{batch_x_mask.shape} batch_y_mask:{batch_y_mask.shape} '\
                #     f'price_batch:{price_batch.shape} gt_batch:{gt_batch.shape}')

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x[0].float().to(self.device)
                batch_y = batch_y[0].float()#.to(self.device)
                batch_x_mark = batch_x_mark[0].float()#.to(self.device)
                batch_y_mark = batch_y_mark[0].float()#.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()#.to(self.device)
                
                x, dec_out, x_mask = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # x       [bs x num_patch x n_vars x patch_len] 输入patch
                # dec_out [bs x num_patch x n_vars x patch_len] 预测patch
                # mask    [bs x num_patch x n_vars]             patch mask
                if epoch == 0 and i == 0:
                    print(f' x {x.shape}\n dec_out {dec_out.shape}\n x_mask {x_mask.shape}\n')
                loss = criterion(x, dec_out, x_mask) # reconstruction_mse_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali_stock(vali_data, vali_loader, criterion)
            test_loss = self.vali_stock(test_data, test_loader, criterion)
            
            train_loss_epoch.append(train_loss)
            val_loss_epoch.append(vali_loss)
            test_loss_epoch.append(test_loss)

            train_loss = np.average(train_loss)
            vali_loss = np.average(vali_loss)
            test_loss = np.average(test_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        np.save(path + '/' + 'train_loss_epoch.npy', train_loss_epoch)
        np.save(path + '/' + 'val_loss_epoch.npy', val_loss_epoch)
        np.save(path + '/' + 'test_loss_epoch.npy', test_loss_epoch)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.load_model(best_model_path)
    
        return self.model


    def transfer_weights(self, weights_path, exclude_head=True):
        new_state_dict = torch.load(weights_path, map_location=self.device)
        # 加上 "module." 前缀 【模型在训练保存时并不保存module】
        if isinstance(self.model, (DDP, nn.DataParallel)):
            state_dict = {}
            for k, v in new_state_dict.items():
                name = "module." + k if not k.startswith("module.") else k
                state_dict[name] = v
            new_state_dict = state_dict

        matched_layers = 0
        unmatched_layers = []
        for name, param in self.model.state_dict().items():  
            if exclude_head and 'head' in name: continue
            if name in new_state_dict:            
                matched_layers += 1
                input_param = new_state_dict[name]
                if input_param.shape == param.shape: param.copy_(input_param)
                else: raise Exception(f"Layer {name} param {param.shape} doesn't macth input_param {input_param.shape}")
            else:
                unmatched_layers.append(name)
                pass # these are weights that weren't in the original model, such as a new head
        
        if matched_layers == 0:
            raise Exception("No shared weight names were found between the models")
        else:
            if len(unmatched_layers) > 0:
                print(f'check unmatched_layers: {unmatched_layers}')
            else:
                print(f"weights from {weights_path} successfully transferred!\n")
        self.model = self.model.to(self.device)
        return self.model

    def get_model(self):
        "Return the model maybe wrapped inside `model`."    
        return self.model.module if isinstance(self.model, (DDP, nn.DataParallel)) else self.model

    def freeze(self):
        """ 
        freeze the model head
        require the model to have head attribute
        """
        if hasattr(self.get_model(), 'head'): 
            print('model head is available')
            for param in self.get_model().parameters(): param.requires_grad = False        
            for param in self.get_model().head.parameters(): param.requires_grad = True
            print('model is frozen except the head')
        
        # 25-1-5 如果微调时需要进行关系建模,则hgat部分也需要进行微调
        if self.args.graph:
            if hasattr(self.get_model(), 'hgat'):
                print('model hgat is availabel')
                for param in self.get_model().hgat.parameters(): param.requires_grad = True
                print('model is frozen except the hgat')

        print('number of model params requires_grad', sum(p.numel() for p in self.get_model().parameters() if p.requires_grad))
        print(f'\n')
    
    def unfreeze(self):
        for param in self.get_model().parameters(): param.requires_grad = True
        print('model is unfrozen')
        print('number of model params requires_grad', sum(p.numel() for p in self.get_model().parameters() if p.requires_grad))
        print(f'\n')

    def finetune(self, setting):
        # load pretrained model
        if self.args.model == 'CI_STHPAN':
            pretrained_model_path = "/home/xiahongjie/UniStock/checkpoints/CI_STHPAN/Stock_maxscale/UTSD-full-npy/pretrain_Stock_maxscale_sl672_it96_CI_STHPAN_Utsd_Npy_Stock_maxscale_ftM_sl672_ll0_pl5_dm1024_nh8_el8_dl1_df2048_graphFalse_a1_q0.01_ne100_hq0.9_Exp_0/checkpoint.pth"
            # if self.args.pretrain_data in ['UTSD', 'Stock_maxscale']:
            #     pretrained_model_path = f'/home/xiahongjie/UniStock/checkpoints/CI_STHPAN/{self.args.pretrain_data}'\
            #                             f'/UTSD-full-npy/pretrain_{self.args.pretrain_data}_sl672_it{self.args.input_token_len}'\
            #                             f'_CI_STHPAN_Utsd_Npy_{self.args.pretrain_data}_ftM_sl672_ll0_pl5_dm1024_nh8_el8_dl1_df2048_graphFalse_a1_q0.01_ne100_hq0.9_Exp_0/checkpoint.pth'
            # else:
            #     raise NotImplementedError
        elif self.args.model == 'Timer':
            pretrained_model_path = "/home/xiahongjie/UniStock/checkpoints/Timer/Stock_maxscale/UTSD-full-npy/pretrain_Stock_maxscale_sl672_it96_Timer_Utsd_Npy_Stock_maxscale_ftM_sl672_ll576_pl5_dm1024_nh8_el8_dl1_df2048_graphFalse_Exp_0/checkpoint.pth"
            # if self.args.pretrain_data == 'UTSD':
            #     pretrained_model_path = os.path.join('./checkpoints', 'Timer', 'Timer_forecast_1.0.pth')
            # elif self.args.pretrain_data in ['Stock', 'A_share', 'S&P500',
            #                                  'Stock_maxscale', 'A_share_maxscale', 'S&P500_maxscale',
            #                                  'Stock_only', 'A_share_only', 'S&P500_only',
            #                                  'Stock_only_maxscale', 'A_share_only_maxscale', 'S&P500_only_maxscale']:
            #     # 25-1-26 由于微调的时候label_len=0, 因此这里需要设置label_len
            #     label_len = self.args.seq_len - self.args.input_token_len
            #     pretrained_model_path = f'/home/xiahongjie/UniStock/checkpoints/Timer/{self.args.pretrain_data}'\
            #                             f'/UTSD-full-npy/pretrain_{self.args.pretrain_data}_sl672_it{self.args.input_token_len}_Timer_Utsd_Npy_'\
            #                             f'{self.args.pretrain_data}_ftM_sl672_ll{label_len}_pl5_dm1024_nh8_el8_dl1_df2048_graphFalse_a1_q0.01_ne100_hq0.9_Exp_0/checkpoint.pth'
            # else:
            #     raise NotImplementedError
        elif self.args.model == 'VisionTS':
            pretrained_model_path = '/home/xiahongjie/xiahongjie_data/Time-Series-Library/checkpoints/visionts/mae_visualize_vit_base.pth'
        elif self.args.model == 'MOMENT':
            pass
        else:
            raise ValueError(f'Error: model {self.args.model} is unknown!!')
        
        if self.args.model in ['VisionTS', 'MOMENT']:
            # 这两个模型都是直接从hugging face加载
            pass
        else:
            print(f'\nfinetune the pretrained model from {pretrained_model_path}')
            self.transfer_weights(pretrained_model_path)
        
        # Finetune the head of freeze_epochs > 0:
        print('\nFinetune the head\n')
        self.freeze()

        path = os.path.join(self.args.checkpoints, setting, self.args.test_begin_date, 'linear_probe')
        if not os.path.exists(path):
            os.makedirs(path)

        if self.args.model == 'MOMENT':
            # MOMENT只微调预测头1个epoch
            self.train(path, train_epochs=1) # linear_probe for 1 epochs
            return

        self.train(path, train_epochs=10) # linear_probe for 10 epochs

        # 25-1-15 由于entire微调反而降低模型效果, 后续不再对模型进行全参数微调
        # Finetune the entire network if n_epochs > 0
        # 25-1/2 如果单独跑全参数微调实验, 应该先加载跑好的linear_prob模型
        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.load_model(best_model_path)

        # print('\nFinetune the entire network\n')
        # self.unfreeze()
        
        # path = os.path.join(self.args.checkpoints, setting, self.args.test_begin_date, 'entire')
        # if not os.path.exists(path):
            # os.makedirs(path)

        # self.train(path, train_epochs=20) # finetune entire network for 20 epochs

    def supervised(self, setting):
        path = os.path.join(self.args.checkpoints, setting, self.args.test_begin_date)
        if not os.path.exists(path):
            os.makedirs(path)
        self.train(path=path, train_epochs=10)

    def train(self, path, train_epochs):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 记录损失
        train_reg_loss_epoch = []
        val_reg_loss_epoch = []
        test_reg_loss_epoch = []

        train_rank_loss_epoch = []
        val_rank_loss_epoch = []
        test_rank_loss_epoch = []

        train_loss_epoch = []
        val_loss_epoch = []
        test_loss_epoch = []

        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []
            train_reg_loss = []
            train_rank_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, 
                    batch_x_mask, batch_y_mask, price_batch, gt_batch) in enumerate(tqdm(train_loader)):
                # print(f'i:{i} batch_x:{batch_x.shape} batch_y:{batch_y.shape} '\
                #     f'batch_x_mark:{batch_x_mark.shape} batch_y_mark:{batch_y_mark.shape} '\
                #     f'batch_x_mask:{batch_x_mask.shape} batch_y_mask:{batch_y_mask.shape} '\
                #     f'price_batch:{price_batch.shape} gt_batch:{gt_batch.shape}')
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x[0].float()
                price_batch = price_batch[0].float()
                gt_batch = gt_batch[0].float()

                # 以下数据未使用
                # batch_y = batch_y[0].float()#.to(self.device)
                # batch_x_mark = batch_x_mark[0].float()#.to(self.device)
                # batch_y_mark = batch_y_mark[0].float()#.to(self.device)
                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()#.to(self.device)

                # 过滤无效数据
                mask = ~torch.isnan(price_batch.squeeze()) # [num_stock] 当日有效交易股票数
                batch_x = batch_x[mask]
                price_batch = price_batch[mask].to(self.device)
                gt_batch = gt_batch[mask].to(self.device)

                num_stocks = batch_x.shape[0]
                if num_stocks == 0:
                    continue
                # print(f'i:{i} valid num_stock:{num_stocks}')

                # 对数据进行随机shuffle, 为了让DyHGAT更好的学习不同股票之间的关系
                if self.args.graph:
                    indices = torch.randperm(num_stocks)
                    batch_x = batch_x[indices]
                    price_batch = price_batch[indices]
                    gt_batch = gt_batch[indices]
                
                # encoder - decoder
                # 24-12/30 这里不能简单对输入样本进行切分, 应该将有效样本传入模型, 随后进行拼接？
                # 24-12/31 但是由于nn.DataParallel会对输入样本进行划分, 因此形成矛盾
                split_steps = num_stocks // self.split_batch
                if num_stocks % self.split_batch != 0:
                    split_steps += 1
                outputs_k = []
                # k循环输入样本数
                for k in range(split_steps):
                    batch_x_k = batch_x[k * self.split_batch : (k + 1) * self.split_batch, :, :].to(self.device)
                    if self.args.model == 'VisionTS':
                        outputs = self.model(batch_x_k, None, None, None)
                    elif self.args.model == 'Timer':
                        outputs, attns = self.model(batch_x_k, None, None, None)
                        # 25-1/2 Timer的输出长度为token_num * token_len, 应该用预测的最后一个token的前pred_len个元素作为模型输出结果
                        outputs = outputs[:, (self.args.token_num - 1) * self.args.output_token_len:, :]
                    elif self.args.model == 'CI_STHPAN':
                        # x_enc, x_mark_enc, x_dec, x_mark_dec
                        # outputs [bs x （token_num * token_len） x nvars] 预测结果
                        outputs, attns = self.model(batch_x_k, None, None, None)
                        outputs = outputs[:, (self.args.token_num - 1) * self.args.output_token_len:, :]
                    elif self.args.model == 'LSTM':
                        outputs = self.model(batch_x_k)
                    elif self.args.model == 'Transformer':
                        outputs = self.model(batch_x_k) 
                    elif self.args.model == 'MOMENT':
                        batch_x_k = batch_x_k.permute(0, 2, 1) # [B, C, L]
                        # print(f'{i} batch_x_k {batch_x_k.shape}') # S&P500 训练会报错
                        outputs = self.model(x_enc=batch_x_k)
                        outputs = outputs.forecast # [B, C, T]
                        outputs = outputs.permute(0, 2, 1) # [B, T, C]
                    elif self.args.model == 'Master':
                        outputs = self.model(x=batch_x_k)
                    elif self.args.model == 'DTML':
                        outputs = self.model(stocks=batch_x_k)
                    else:
                        raise NotImplementedError

                    if epoch == 0 and i == 0:
                        print(f'k {k} batch_x {batch_x.shape} split_batch {self.split_batch} batch_x_k {batch_x_k.shape} outputs {outputs.shape}')
                    outputs_k.append(outputs)

                    # torch.cuda.empty_cache()
                outputs = torch.cat(outputs_k, dim=0)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                # 24-12/23 这里决定了微调阶段只用模型的多少长度的输出结果
                # 25-1/2 取前pred_len个预测值 [这里是和time series libary目的不同的地方]
                outputs = outputs[:, :self.args.pred_len, f_dim:]
                if epoch == 0 and i == 0:
                    print(f' batch_x {batch_x.shape}\n outputs {outputs.shape}\n')
                
                loss, reg_loss, rank_loss, rr = criterion(i, outputs, price_batch, gt_batch,
                                                            self.args.alpha, self.device)
                train_reg_loss.append(reg_loss.item()) # tensor.item() 只能用于包含单个元素的张量（即标量）
                train_rank_loss.append(rank_loss.item())
                train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            if self.args.model == 'MOMENT':
                vali_loss, vali_reg_loss, vali_rank_loss = np.nan, np.nan, np.nan
            else:
                vali_loss, vali_reg_loss, vali_rank_loss = self.vali_stock(vali_data, vali_loader, criterion)
                # test_loss, test_reg_loss, test_rank_loss = self.vali_stock(test_data, test_loader, criterion)

            train_reg_loss_epoch.append(train_reg_loss)
            val_reg_loss_epoch.append(vali_reg_loss)
            # test_reg_loss_epoch.append(test_reg_loss)

            train_rank_loss_epoch.append(train_rank_loss)
            val_rank_loss_epoch.append(vali_rank_loss)
            # test_rank_loss_epoch.append(test_rank_loss)

            train_loss_epoch.append(train_loss)
            val_loss_epoch.append(vali_loss)
            # test_loss_epoch.append(test_loss)

            # 打印loss均值
            train_reg_loss = np.average(train_reg_loss)
            train_rank_loss = np.average(train_rank_loss)
            train_loss = np.average(train_loss)

            vali_reg_loss = np.average(vali_reg_loss)
            vali_rank_loss = np.average(vali_rank_loss)
            vali_loss = np.average(vali_loss)

            test_reg_loss = np.nan #np.average(test_reg_loss)
            test_rank_loss = np.nan #np.average(test_rank_loss)
            test_loss = np.nan #np.average(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Reg Loss: {2:.7f} Vali Reg Loss: {3:.7f} Test Reg Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_reg_loss, vali_reg_loss, test_reg_loss))
            print("Epoch: {0}, Steps: {1} | Train Rank Loss: {2:.7f} Vali Rank Loss: {3:.7f} Test Rank Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_rank_loss, vali_rank_loss, test_rank_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        torch.cuda.empty_cache()
        best_model_path = path + '/' + 'checkpoint.pth'
        self.load_model(best_model_path)

        # loss save
        np.save(path + '/' + 'train_reg_loss_epoch.npy', train_reg_loss_epoch)
        np.save(path + '/' + 'val_reg_loss_epoch.npy', val_reg_loss_epoch)
        # np.save(path + '/' + 'test_reg_loss_epoch.npy', test_reg_loss_epoch)

        np.save(path + '/' + 'train_rank_loss_epoch.npy', train_rank_loss_epoch)
        np.save(path + '/' + 'val_rank_loss_epoch.npy', val_rank_loss_epoch)
        # np.save(path + '/' + 'test_rank_loss_epoch.npy', test_rank_loss_epoch)

        np.save(path + '/' + 'train_loss_epoch.npy', train_loss_epoch)
        np.save(path + '/' + 'val_loss_epoch.npy', val_loss_epoch)
        # np.save(path + '/' + 'test_loss_epoch.npy', test_loss_epoch)

        return self.model
    

    def load_model(self, weight_path):
        state_dict = torch.load(weight_path)

        if isinstance(self.model, (DDP, nn.DataParallel)):
            new_state_dict = {}
            for k, v in state_dict.items():
                # 加上 "module." 前缀 【模型在训练保存时并不保存module】
                name = "module." + k if not k.startswith("module.") else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        self.model.load_state_dict(state_dict)
        print(f'loading model from {weight_path} successfully!')

    def test(self, setting, weight_path=None):
        '''
            args.output_len和args.pred_len的区别
                args.output_len为模型进行预测时候的总预测长度
                args.pred_len为模型进行一次预测能预测的长度
            args.output_token_len和args.pred_len的区别
                由于在next-token-prediction自回归训练中, 模型一次预测的长度为output_token_len
                因此args.pred_len = args.output_token_len
        '''
        # 加载模型参数
        if weight_path != None:
            self.load_model(weight_path)
        else:
            print(f'Warning! test without weight_path')

        if self.args.model == 'TimeMOE':
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                "/home/xiahongjie/UniStock/checkpoints/TimeMOE/50M",
                device_map="cuda",
                trust_remote_code=True,
            )
            print('number of model params', sum(p.numel() for p in self.model.parameters()))

        test_data, test_loader  = self._get_data(flag='test')

        inference_steps = self.args.output_len // self.args.pred_len
        dis = self.args.output_len - inference_steps * self.args.pred_len
        if dis != 0:
            inference_steps += 1
        print(f'pred_len = {self.args.pred_len} output_len = {self.args.output_len} inference_steps = {inference_steps} dis = {dis}')

        self.batch_size = test_data.batch_size # total_stock_num
        results = np.zeros([self.batch_size, len(test_loader), self.args.output_len], dtype=float)  # 预测收盘价
        preds = np.zeros([self.batch_size, len(test_loader), self.args.output_len], dtype=float)    # 预测收益率
        trues = np.zeros([self.batch_size, len(test_loader), self.args.output_len], dtype=float)    # 真实收益率

        self.model.eval()
        with torch.no_grad():
            # i 循环len(test_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, 
                    batch_x_mask, batch_y_mask, price_batch, gt_batch) in enumerate(tqdm(test_loader)):
                batch_x = batch_x[0].float()
                price_batch = price_batch[0].float()
                gt_batch = gt_batch[0].float()

                # 以下数据未使用
                # batch_y = batch_y[0].float()#.to(self.device)
                # batch_x_mark = batch_x_mark[0].float()#.to(self.device)
                # batch_y_mark = batch_y_mark[0].float()#.to(self.device)
                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()#.to(self.device)

                # 过滤无效数据
                mask = ~torch.isnan(price_batch.squeeze()) # [valid_num_stock] 当日有效交易股票数
                mask_nan = torch.isnan(price_batch.squeeze())

                batch_x = batch_x[mask]
                price_batch = price_batch[mask]
                gt_batch = gt_batch[mask]
                num_stocks = batch_x.shape[0]
                # print(f'i {i} num_stocks {num_stocks}')
                
                # 无效股票设置为nan
                mask_nan = mask_nan.cpu().numpy()
                results[mask_nan, i, :] = np.nan
                preds[mask_nan, i, :] = np.nan
                trues[mask_nan, i, :] = np.nan

                pred_y = []
                # j 循环output_len
                for j in range(inference_steps):                    
                    # 24-12/24 考虑到截面个股数量较多导致模型训练/推理占用显存较大, 这里对输入数据沿着第1个维度进行切分循环
                    split_steps = num_stocks // self.split_batch
                    if num_stocks % self.split_batch != 0:
                        split_steps += 1
                    
                    output_y_k = []
                    # k循环输入样本数
                    for k in range(split_steps):
                        # [batch_size, seq_len, n_vars]
                        batch_x_k = batch_x[k * self.split_batch : (k + 1) * self.split_batch, :, :].to(self.device)
                        if j:
                            # autoregresive
                            pred_y_k = pred_y[-1][k * self.split_batch : (k + 1) * self.split_batch, :, :].to(self.device)
                            batch_x_k = torch.cat([batch_x_k[:, self.args.pred_len:, :], pred_y_k], dim=1)
                        
                        if self.args.model == 'VisionTS':
                            outputs = self.model(batch_x_k, i, None, None)
                        elif self.args.model == 'Timer':
                            outputs, attns = self.model(batch_x_k, None, None, None)
                            # 25-1/2 取最后一个token
                            outputs = outputs[:, (self.args.token_num - 1) * self.args.output_token_len:, :]
                        elif self.args.model == 'CI_STHPAN':
                            outputs, attns = self.model(batch_x_k, None, None, None)
                            # 25-1/15 取最后一个token
                            outputs = outputs[:, (self.args.token_num - 1) * self.args.output_token_len:, :]
                        elif self.args.model == 'LSTM':
                            outputs = self.model(batch_x_k)
                        elif self.args.model == 'TimeMOE':
                            normed_seqs = batch_x_k.squeeze(-1) 
                            output = self.model.generate(normed_seqs, max_new_tokens=5)
                            normed_predictions = output[:, -5:]
                            outputs = normed_predictions.unsqueeze(-1)
                        elif self.args.model == 'Transformer':
                            outputs = self.model(batch_x_k) 
                        elif self.args.model == 'TimesFM':
                            # input_ts, input_padding, freq
                            outputs = self.model(batch_x_k, None, None)
                        elif self.args.model == 'MOMENT':
                            batch_x_k = batch_x_k.permute(0, 2, 1) # [B, C, L]
                            outputs = self.model(x_enc=batch_x_k)
                            outputs = outputs.forecast # [B, C, T]
                            outputs = outputs.permute(0, 2, 1) # [B, T, C]
                        elif self.args.model == 'Moirai':
                            past_observed_target = torch.ones_like(batch_x_k, dtype=torch.bool).to(self.device)
                            past_is_pad = torch.zeros_like(batch_x_k, dtype=torch.bool).squeeze(-1).to(self.device)
                            outputs = self.model(
                                past_target=batch_x_k,
                                past_observed_target=past_observed_target,
                                past_is_pad=past_is_pad,
                            ) # [B, num_samples, pred_len]
                            outputs = torch.mean(outputs, dim=1, keepdim=True)  # [B, 1, pred_len]
                            outputs = outputs.permute(0, 2, 1) # [B, pred_len, 1]
                        elif self.args.model == 'Master':
                            outputs = self.model(x=batch_x_k)
                        elif self.args.model == 'DTML':
                            outputs = self.model(stocks=batch_x_k)
                        else:
                            raise NotImplementedError

                        if i == 0:
                            print(f'j {j} k {k} batch_x {batch_x.shape} split_batch {self.split_batch} batch_x_k {batch_x_k.shape} outputs {outputs.shape}')
                        # 25-1/2 取前pred_len个预测值
                        outputs = outputs[:, :self.args.pred_len, :]
                        output_y_k.append(outputs.detach().cpu())
                        # 25-1-6 每次模型跑完数据后,都应该及时清除显存,这对于graph尤其重要,但是会明显变慢
                        torch.cuda.empty_cache()
                    output_y_k = torch.cat(output_y_k, dim=0)
                    pred_y.append(output_y_k)
                
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-(self.args.pred_len - dis), :]
                gt_batch = gt_batch[:, -self.args.output_len:]

                f_dim = -1 if self.args.features == 'MS' else 0
                pred_y = pred_y[:, :, f_dim:]
                pred_y = pred_y.reshape((num_stocks, self.args.output_len))

                pred_y = pred_y.cpu().numpy()
                price_batch = price_batch.cpu().numpy()
                gt_batch = gt_batch.cpu().numpy()

                rr = np.divide((pred_y - price_batch), price_batch)

                if i == 0:
                    print(f"pred_y shape: {pred_y.shape} rr shape: {rr.shape} gt_batch shape: {gt_batch.shape}")
                mask = mask.cpu().numpy()
                results[mask, i, :] = pred_y        # [num_stocks, output_len]
                preds[mask, i, :] = rr              # [num_stocks, output_len]
                trues[mask, i, :] = gt_batch       # [num_stocks, output_len]
        torch.cuda.empty_cache()
        print(f'\nTest Done:')
        print(f"results shape: {results.shape} preds shape: {preds.shape} trues shape: {trues.shape}")
        
        if self.args.model == 'LSTM' or self.args.model == 'GRU' or self.args.model == 'ALSTM' \
                            or self.args.model == 'GAT' or self.args.model == 'Transformer':
            folder_path = os.path.join('./test_results', self.args.model, self.args.market_name, self.args.data_path, setting, str(self.args.alpha), self.args.test_begin_date)
        else:
            folder_path = os.path.join('./test_results', self.args.model, self.args.market_name, self.args.data_path, self.args.pretrain_data, setting, self.args.test_begin_date)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 24-11/21 新增指标
        mae, mse, rmse, mape, mspe, acc, perf = backtest(preds, trues, folder_path, self.args.market_name)
        print(f'Test MSE: {mse:.4f} MAE: {mae:.4f} RMSE: {rmse:.4f} MAPE: {mape:.4f} MSPE: {mspe:.4f} ACC: {acc:.2%} preformance:\n{perf}\n')
        f = open("result_stock.txt", 'a')
        f.write(folder_path + "  \n")
        f.write(f'\nTest MSE: {mse:.4f} MAE: {mae:.4f} RMSE: {rmse:.4f} MAPE: {mape:.4f} MSPE: {mspe:.4f} ACC: {acc:.2%} preformance:\n{perf}\n\n')
        f.close()
        
        # np.save(folder_path + '/' + 'pred.npy', preds)
        # np.save(folder_path + '/' + 'true.npy', trues)
        # np.save(folder_path + '/' + 'result.npy', results)   # 24-11-26 result为模型的outputs, 没有进行逆标准化
