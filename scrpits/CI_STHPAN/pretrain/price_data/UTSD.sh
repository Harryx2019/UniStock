# export CUDA_VISIBLE_DEVICES=1,2,3

# basic config
model_name=CI_STHPAN
task_name=pretrain
data=Utsd_Npy
market_name=UTSD
root_path=/home/xiahongjie/UniStock/dataset
data_path=UTSD-full-npy
# ['open', 'high', 'low', 'close', 'volume']
# pretrain 全部视为单变量进行训练
enc_in=1
seed=2025

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


# dataset config
token_num=7
input_token_len=96
output_token_len=96
seq_len=$[$token_num*$input_token_len]
label_len=0
batch_size=8192
stride_dataset=1


#######################下述参数与预训练关系不大###########
pred_len=5
output_len=10
# train_begin_date参数对data_stock有重要意义, 决定了数据标准化的情况
train_begin_date='1990-01-01'
valid_begin_date='2019-01-01'
test_begin_date='2020-01-01'
test_end_date='2021-01-01'
######################################################

# model config
model_id_name=$market_name'_sl'$seq_len'_it'$input_token_len
checkpoints=./checkpoints/$model_name/$market_name/$data_path

e_layers=8
d_model=1024
d_ff=2048
n_heads=8

python -u run.py \
  --random_seed $seed \
  --task_name $task_name \
  --is_training 1 \
  --train_epochs 100 \
  --patience 3 \
  --checkpoints $checkpoints \
  --root_path $root_path \
  --data_path $data_path \
  --scale \
  --market_name $market_name \
  --model_id $model_id_name \
  --model $model_name \
  --data $data \
  --target close \
  --features M \
  --freq d \
  --stride_dataset $stride_dataset \
  --input_token_len $input_token_len \
  --output_token_len $output_token_len \
  --token_num $token_num \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $pred_len \
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
  --use_multi_gpu \
  --des 'Exp' \
  --batch_size $batch_size \
  --itr 1 >logs/$model_name/$task_name/$data_path/$market_name'_sl'$seq_len'_it'$input_token_len'_ol'$output_token_len'_el'$e_layers'_dm'$d_model'_nh'$n_heads.log