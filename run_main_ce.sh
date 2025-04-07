#!/bin/bash

# Lista de modelos a testar
models=("resnet50")

# Hiperparâmetros fixos
BATCH_SIZE=32
SIZE=128
LR=0.0001
num_workers=8
N_CLS=10
EPOCHS=1
EXTRA_ARGS="--cosine"

# Loop pelos modelos
for model in "${models[@]}"
do
    echo "Iniciando treinamento para modelo: $model"
    nohup time pipenv run python main_ce.py \
        --batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --epochs $EPOCHS \
        --n_cls $N_CLS \
        --model $model \
        --dataset "path" \
        --size $SIZE \
        --num_workers $num_workers \
        $EXTRA_ARGS > "CE_${model}_.txt" &
done

echo "Todos os experimentos foram iniciados em background."
