from data_provider.data_stock import Dataset_Stock
from data_provider.data_UTSD_Npy import UTSD_Npy
from torch.utils.data import DataLoader


data_dict = {
    'Stock': Dataset_Stock,
    'Utsd_Npy': UTSD_Npy
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    
    shuffle_flag = False if flag == 'test' else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq
    
    if args.task_name == 'pretrain' or args.task_name == 'finetune' or args.task_name == 'zeroshot' or args.task_name == 'supervised':
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            market_name=args.market_name,
            flag=flag,
            input_token_len=args.input_token_len,
            output_token_len=args.output_token_len,
            autoregressive=args.autoregressive,
            size=[args.seq_len, args.label_len, args.output_len] if flag == 'test' else [args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=freq,
            stride=args.stride_dataset
        )
        print(f'{flag} {len(data_set)}\n')
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last
        )

        return data_set, data_loader
    else:
        raise NotImplementedError
