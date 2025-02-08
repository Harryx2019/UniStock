# export CUDA_VISIBLE_DEVICES=6
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
market_name=NASDAQ
data_path=price_data
enc_in=4

# dataset config
token_num=56
input_token_len=12
output_token_len=12
seq_len=$[$token_num*$input_token_len]

pred_len=5
output_len=10

# model config
model_id_name=$market_name'_sl'$seq_len'_pl'$pred_len'_it'$input_token_len
checkpoints=./checkpoints/$model_name/$market_name/$data_path

e_layers=3
d_model=128
d_ff=256
n_heads=8

alpha=1

python -u run.py \
  --task_name $task_name \
  --is_training 0 \
  --checkpoints $checkpoints \
  --root_path ~/xiahongjie_data/dataset \
  --data_path $data_path \
  --scale \
  --market_name $market_name \
  --model_id $model_id_name \
  --model $model_name \
  --data Stock \
  --target close \
  --features MS \
  --freq d \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --output_len $output_len \
  --patch_len $input_token_len \
  --stride $input_token_len \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --enc_in $enc_in \
  --d_model $d_model \
  --d_ff $d_ff \
  --alpha $alpha \
  --use_multi_gpu \
  --des 'Exp' \
  --batch_size 1 \
  --itr 1 >logs/CI-STHPAN/test/price_data/$market_name'_sl'$seq_len'_it'$input_token_len'_pl'$pred_len'_ol'$output_len'_el'$e_layers'_dm'$d_model'_nh'$n_heads'_a'$alpha.log