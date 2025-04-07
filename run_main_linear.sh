#!/bin/bash

checkpoints=(
    
    #"save_exp_subfolds/save_2025_03_31/SINCERE/path_models/SINCERE_path_resnet50_lr_0.0001_decay_0.0001_bsz_128_temp_0.07_trial_0_cosine_warm_2025_03_31-20_58_46/last_fold_1.pth"
    #"save_2025_04_02/SINCERE/path_models/SINCERE_path_vit_small_lr_0.0001_decay_0.0001_bsz_128_temp_0.07_trial_0_cosine_warm_2025_04_02-11_52_52/last_fold_1.pth"
    "save_exp_subfolds/save_2025_04_02/SINCERE/path_models/SINCERE_path_vit_small_lr_0.0001_decay_0.0001_bsz_128_temp_0.07_trial_0_cosine_warm_2025_04_02-11_52_52/last_fold_1.pth"
)

# Parâmetros gerais do comando
batch_size=32
n_cls=10
size=128
learning_rate=0.0001
dataset="path"
num_workers=8
model="vit_small"

# Loop para executar a avaliação para cada checkpoint
for ckpt in "${checkpoints[@]}"; do
    # Nome do método extraído do caminho do checkpoint (opcional para fins de log)
    method_name=$(basename $(dirname $ckpt) | cut -d'_' -f1)

    echo "Iniciando avaliação linear para o checkpoint: $ckpt"
    time python main_linear.py \
        --batch_size $batch_size \
        --n_cls $n_cls \
        --size $size \
        --model $model \
        --dataset $dataset \
        --learning_rate $learning_rate \
        --num_workers $num_workers \
        --epochs 2 \
        --ckpt $ckpt \
        > "${method_name}_LinearEvalStage-${model}.txt"

    echo "Avaliação para o checkpoint $ckpt finalizada."
done
                                