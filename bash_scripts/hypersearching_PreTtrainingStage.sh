#!/bin/bash

dia="27_abril2025_vit_warm_lr"

# Parâmetros gerais
NAME_DATASET="ModelNet10"
dataset="path"

# Arrays com diferentes hiperparâmetros para teste
batch_sizes=(128 256)
learning_rates=(0.5 0.001 0.0001 0.00001)
sizes=(224)  # Variando o tamanho da imagem

# Modelos e métodos
declare -a models=(
   "vit_small"
   "resnet50" 
  # "vit_base"
    #"dino_vit_small_p_16"
  # "dino_vit_small_p_8"
   # "dino_vit_base_p_16"
  # "dino_vit_base_p_8"
  # "dinov2_vit_small_p_14"
  # "dinov2_vit_base_p_14"
)

declare -a methods=(
   "SupCon"
   "SINCERE"
   # "EpsSupInfoNCE"
   #"InfoNCE"
   #"SimCLR"
)

# Diretório de logs (sem timestamp)
log_dir="experiments/${dia}"
mkdir -p "$log_dir"

# Loop para rodar todas as combinações de hiperparâmetros
for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for learning_rate in "${learning_rates[@]}"; do
                for size in "${sizes[@]}"; do
                    echo "Iniciando treinamento: Modelo=$model | Loss=$method | Batch=$batch_size | LR=$learning_rate | Size=$size"

                    # Adiciona --warm se o modelo for resnet50
                    warm_flag=""
		    if [ "$model" == "resnet50" ] || [ "$method" == "SupCon" ]; then
                        warm_flag="--warm"
                    fi

                    time python main_supcon.py \
                        --batch_size $batch_size \
                        --size $size \
                        --model $model \
                        --epochs 100 \
                        --learning_rate $learning_rate \
                        --dataset $dataset \
                        --train_mode cross-validation \
                        --num_folds 4 \
                        --train_file Datasets/train_new_subfold.csv \
                        --num_workers 8 \
                        --method $method \
                        --cosine \
                        $warm_flag\
                        > "$log_dir/${NAME_DATASET}_${model}_${method}_B${batch_size}_LR${learning_rate}_S${size}.txt"

                    echo "Finalizado: Modelo=$model | Loss=$method | Batch=$batch_size | LR=$learning_rate | Size=$size\n"
                done
            done
        done
    done
done

echo "Todos os treinamentos foram concluídos."

