export CUDA_VISIBLE_DEVICES=1,2,3

# Tips:
# 1. 对预训练模型微调10个epoch head层, 20个epoch entire network, 因此不需要设置train_epochs/patience
# 2. 对于Stock数据集, price_data为5个特征[open, high, low, close, volume], 且需要scale&inverse
# 3. fine-tune阶段受多gpu训练影响, 需要使得gpu个数为stock_num的约数 (TODO:待验证？)
# 4. fine-tune阶段训练目标为reg_loss + alpha * rank_loss
# 5. 微调阶段需要确认模型的预测能力, 即pred_len, 同时由于微调阶段需要加载测试集, 因此要保证output_len = pred_len
# 6. fine-tune阶段需要保证模型模式为MS, 因为只需要预测close


# basic config
model_name=Timer
task_name=finetune
data=Stock
market_name=A_share
root_path=/home/xiahongjie/UniStock/dataset/stock
data_path=price_data
enc_in=1
seed=2025

# pretrain data
# pretrain_data=UTSD
# pretrain_data=A_share
# pretrain_data=A_share_maxscale
pretrain_data=Stock_maxscale
# pretrain_data=Stock_only
# pretrain_data=A_share_only

# build log file
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

if [ ! -d "./logs/$model_name/$task_name" ]; then
    mkdir ./logs/$model_name/$task_name
fi

if [ ! -d "./logs/$model_name/$task_name/$data_path" ]; then
    mkdir ./logs/$model_name/$task_name/$data_path
fi

if [ ! -d "./logs/$model_name/$task_name/$data_path/$pretrain_data" ]; then
    mkdir ./logs/$model_name/$task_name/$data_path/$pretrain_data
fi

token_num=7
input_token_len=96
output_token_len=96
seq_len=$[$token_num*$input_token_len]

# dataset config
pred_len=5
output_len=10

quantile=0.05
stride_dataset=1

# train_begin_date参数对data_stock有重要意义, 决定了数据标准化的情况
# train_begin_date=('2016-01-01' '2017-01-01' '2018-01-01' '2019-01-01' '2020-01-01')
# valid_begin_date=('2019-01-01' '2020-01-01' '2021-01-01' '2022-01-01' '2023-01-01')
# test_begin_date=('2020-01-01' '2021-01-01' '2022-01-01' '2023-01-01' '2024-01-01')
# test_end_date=('2021-01-01' '2022-01-01' '2023-01-01' '2024-01-01' '2025-01-01')

train_begin_date=('2018-01-01')
valid_begin_date=('2021-01-01')
test_begin_date=('2022-01-01')
test_end_date=('2023-01-01')

# model config
model_id_name=$market_name'_sl'$seq_len'_pl'$pred_len'_it'$input_token_len
checkpoints=./checkpoints/$model_name/$market_name/$data_path/$pretrain_data

e_layers=8
d_model=1024
d_ff=2048
n_heads=8

alpha=1

# DyHGAT config
graph=1
num_hyperedges=200
hyperedges_quantile=0.9

for i in "${!train_begin_date[@]}"; do
    python -u run.py \
    --random_seed $seed \
    --task_name $task_name \
    --is_training 1 \
    --checkpoints $checkpoints \
    --root_path $root_path \
    --data_path $data_path \
    --pretrain_data $pretrain_data \
    --scale \
    --market_name $market_name \
    --model_id $model_id_name \
    --model $model_name \
    --data $data \
    --target close \
    --features MS \
    --freq d \
    --stride_dataset $stride_dataset \
    --input_token_len $input_token_len \
    --output_token_len $output_token_len \
    --token_num $token_num \
    --quantile $quantile \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --output_len $output_len \
    --patch_len $input_token_len \
    --stride $input_token_len \
    --train_begin_date ${train_begin_date[$i]} \
    --valid_begin_date ${valid_begin_date[$i]} \
    --test_begin_date ${test_begin_date[$i]} \
    --test_end_date ${test_end_date[$i]} \
    --e_layers $e_layers \
    --n_heads $n_heads \
    --enc_in $enc_in \
    --d_model $d_model \
    --d_ff $d_ff \
    --alpha $alpha \
    --use_multi_gpu \
    --devices 0,1,2 \
    --des 'Exp' \
    --batch_size 1 \
    --itr 1 >logs/$model_name/$task_name/$data_path/$pretrain_data/$market_name'_'${test_begin_date[$i]}'_sl'$seq_len'_it'$input_token_len'_pl'$pred_len'_el'$e_layers'_dm'$d_model'_nh'$n_heads'_a'$alpha'_graph'$graph'_q'$quantile'_ne'$num_hyperedges'_hq'$hyperedges_quantile.log
done