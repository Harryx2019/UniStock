# export CUDA_VISIBLE_DEVICES=6
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/CI-STHPAN" ]; then
    mkdir ./logs/CI-STHPAN
fi

if [ ! -d "./logs/CI-STHPAN/pretrain" ]; then
    mkdir ./logs/CI-STHPAN/pretrain
fi

if [ ! -d "./logs/CI-STHPAN/pretrain/features" ]; then
    mkdir ./logs/CI-STHPAN/pretrain/features
fi

# basic config
model_name=CI_STHPAN
task_name=pretrain
market_name=NASDAQ
data_path=features
enc_in=5
seed=2025

# dataset config
token_num=56
input_token_len=12
output_token_len=12
seq_len=$[$token_num*$input_token_len]

pred_len=5
output_len=10

# model config
model_id_name=$market_name'_sl'$seq_len
checkpoints=./checkpoints/$model_name/$market_name/$data_path

e_layers=3
d_model=128
d_ff=256
n_heads=8

python -u run.py \
  --random_seed $seed \
  --task_name $task_name \
  --is_training 1 \
  --train_epochs 100 \
  --patience 3 \
  --checkpoints $checkpoints \
  --root_path ~/xiahongjie_data/dataset \
  --data_path $data_path \
  --market_name $market_name \
  --model_id $model_id_name \
  --model $model_name \
  --data Stock \
  --target close \
  --features M \
  --freq d \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --patch_len $input_token_len \
  --stride $input_token_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --enc_in $enc_in \
  --d_model $d_model \
  --d_ff $d_ff \
  --use_multi_gpu \
  --des 'Exp' \
  --batch_size 1 \
  --itr 1 >logs/CI-STHPAN/pretrain/features/$market_name'_sl'$seq_len'_it'$input_token_len'_el'$e_layers'_dm'$d_model'_nh'$n_heads.log