export CUDA_VISIBLE_DEVICES=1,2
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/CI-STHPAN" ]; then
    mkdir ./logs/CI-STHPAN
fi

if [ ! -d "./logs/CI-STHPAN/test" ]; then
    mkdir ./logs/CI-STHPAN/test
fi

if [ ! -d "./logs/CI-STHPAN/test/price_data" ]; then
    mkdir ./logs/CI-STHPAN/test/price_data
fi

# Tips:
# test测试的是微调后的模型

# basic config
model_name=CI_STHPAN
task_name=finetune
data=Stock
market_name=A_share
root_path=/home/xiahongjie/UniStock/dataset/stock
# price_data需要进行max标准化
data_path=price_data
# ['open', 'high', 'low', 'close', 'volume']
enc_in=5

# dataset config
token_num=56
input_token_len=12
output_token_len=12
seq_len=$[$token_num*$input_token_len]

pred_len=5
output_len=10

quantile=0.01

# train_begin_date参数对data_stock有重要意义, 决定了数据标准化的情况
train_begin_date='1990-01-01'
valid_begin_date='2019-01-01'
test_begin_date='2024-01-01'
test_end_date='2025-01-01'

# model config
model_id_name=$market_name'_sl'$seq_len'_pl'$pred_len'_it'$input_token_len
checkpoints=./checkpoints/$model_name/$market_name/$data_path

e_layers=8
d_model=1024
d_ff=2048
n_heads=8

alpha=1

python -u run.py \
  --task_name $task_name \
  --is_training 0 \
  --checkpoints $checkpoints \
  --root_path $root_path \
  --data_path $data_path \
  --scale \
  --market_name $market_name \
  --model_id $model_id_name \
  --model $model_name \
  --data $data \
  --target close \
  --features MS \
  --freq d \
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
  --train_begin_date $train_begin_date \
  --valid_begin_date $valid_begin_date \
  --test_begin_date $test_begin_date \
  --test_end_date $test_end_date \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --enc_in $enc_in \
  --d_model $d_model \
  --d_ff $d_ff \
  --alpha $alpha \
  --use_multi_gpu \
  --devices 0,1 \
  --des 'Exp' \
  --batch_size 1 \
  --itr 1 >logs/CI-STHPAN/test/price_data/$market_name'_'$test_begin_date'_sl'$seq_len'_it'$input_token_len'_pl'$pred_len'_ol'$output_len'_el'$e_layers'_dm'$d_model'_nh'$n_heads'_a'$alpha.log