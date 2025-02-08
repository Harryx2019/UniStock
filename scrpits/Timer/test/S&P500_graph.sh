export CUDA_VISIBLE_DEVICES=1,2,3
# Tips:
# test测试的是微调后的模型

# basic config
model_name=Timer
task_name=finetune
data=Stock
market_name='S&P500'
root_path=/home/xiahongjie/UniStock/dataset/stock
# price_data需要进行max标准化
data_path=price_data
enc_in=1

# pretrain data
# pretrain_data=UTSD
pretrain_data='S&P500_only_maxscale'

# build log file
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

if [ ! -d "./logs/$model_name/test" ]; then
    mkdir ./logs/$model_name/test
fi

if [ ! -d "./logs/$model_name/test/$data_path" ]; then
    mkdir ./logs/$model_name/test/$data_path
fi

if [ ! -d "./logs/$model_name/test/$data_path/$pretrain_data" ]; then
    mkdir ./logs/$model_name/test/$data_path/$pretrain_data
fi


# dataset config
token_num=7
input_token_len=96
output_token_len=96
seq_len=$[$token_num*$input_token_len]

pred_len=5
output_len=10

quantile=0.01
stride_dataset=1

# train_begin_date参数对data_stock有重要意义, 决定了数据标准化的情况
train_begin_date=('2016-07-06' '2017-01-01' '2018-01-01' '2019-01-01')
valid_begin_date=('2019-07-06' '2020-01-01' '2021-01-01' '2022-01-01')
test_begin_date=('2020-07-06' '2021-01-01' '2022-01-01' '2023-01-01')
test_end_date=('2021-01-01' '2022-01-01' '2023-01-01' '2024-01-01')

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
num_hyperedges=100
# 这个参数对显存占用的影响非常大！
hyperedges_quantile=0.9

for i in "${!train_begin_date[@]}"; do
    python -u run.py \
    --task_name $task_name \
    --is_training 0 \
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
    --graph \
    --num_hyperedges $num_hyperedges \
    --hyperedges_quantile $hyperedges_quantile \
    --use_multi_gpu \
    --devices 0,1,2 \
    --des 'Exp' \
    --batch_size 1 \
    --itr 1 >logs/$model_name/test/$data_path/$pretrain_data/$market_name'_'${test_begin_date[$i]}'_sl'$seq_len'_it'$input_token_len'_pl'$pred_len'_ol'$output_len'_el'$e_layers'_dm'$d_model'_nh'$n_heads'_a'$alpha'_graph'$graph.log
done