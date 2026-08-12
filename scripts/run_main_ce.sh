#!/bin/bash

NAME_DATASET="ModelNet40"
dataset_train="Datasets/train_M40.csv"
dataset_test="Datasets/test_M40.csv"
dataset_path="path"
n_cls=40

# Lista de modelos a testar
models=(
    "resnet50"
    "vit_small"
    "vit_base"
    "dino_vit_small_p_16"
    "dino_vit_base_p_16"
)


# Hiperparâmetros fixos
BATCH_SIZE=128
SIZE=224
LR=0.0001
num_workers=16
EPOCHS=100


# Loop pelos modelos
for model in "${models[@]}"; do
    echo "Iniciando treinamento para modelo: $model"
    time python main_ce.py \
        --batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --epochs $EPOCHS \
        --n_cls $n_cls \
        --model $model \
        --dataset $dataset_path \
        --train_file $dataset_train \
        --test_file $dataset_test \
        --size $SIZE \
        --num_workers $num_workers \
       	> "paper_experiments/CrossEntropy_${NAME_DATASET}/${NAME_DATASET}_CE_${model}.txt" 
done

echo "Todos os experimentos foram iniciados em background."
