# export CUDA_VISIBLE_DEVICES=6
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/CI-STHPAN" ]; then
    mkdir ./logs/CI-STHPAN
fi

if [ ! -d "./logs/CI-STHPAN/finetune" ]; then
    mkdir ./logs/CI-STHPAN/finetune
fi

if [ ! -d "./logs/CI-STHPAN/finetune/features" ]; then
    mkdir ./logs/CI-STHPAN/finetune/features
fi

# Tips:
# 1. 对预训练模型微调10个epoch head层, 20个epoch entire network, 因此不需要设置train_epochs/patience
# 2. 对于NASDAQ/NYSE数据集, features为5个特征[ma_5, ma_10, ma_20, ma_30, close], price_data为4个特征[open, high, low, close], 且需要scale&inverse
# 3. fine-tune阶段受多gpu训练影响, 需要使得gpu个数为stock_num的约数
# 4. fine-tune阶段训练目标为reg_loss + alpha * rank_loss
# 5. 微调阶段需要确认模型的预测能力, 即pred_len

# basic config
model_name=CI_STHPAN
task_name=finetune
market_name=NYSE
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
model_id_name=$market_name'_sl'$seq_len'_pl'$pred_len'_it'$input_token_len
checkpoints=./checkpoints/$model_name/$market_name/$data_path

e_layers=3
d_model=128
d_ff=256
n_heads=8

alpha=1

python -u run.py \
  --random_seed $seed \
  --task_name $task_name \
  --is_training 1 \
  --checkpoints $checkpoints \
  --root_path ~/xiahongjie_data/dataset \
  --data_path $data_path \
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
  --patch_len $input_token_len \
  --stride $input_token_len \
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
  --itr 1 >logs/CI-STHPAN/finetune/features/$market_name'_sl'$seq_len'_it'$input_token_len'_pl'$pred_len'_el'$e_layers'_dm'$d_model'_nh'$n_heads'_a'$alpha.log