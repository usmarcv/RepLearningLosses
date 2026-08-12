#!/bin/bash

checkpoints=(
	#"save/Contrastive/InfoNCE/path_models/path_dino_vit_small_p_16_InfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_14-05_20_49/last.pth"
	#"save/Contrastive/SimCLR/path_models/path_dino_vit_small_p_16_SimCLR_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_14-09_00_49/last.pth"
	#"save/Contrastive/EpsSupInfoNCE/path_models/path_dino_vit_small_p_16_EpsSupInfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_14-01_41_09/last.pth"
	#"save/Contrastive/SINCERE/path_models/path_dino_vit_small_p_16_SINCERE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_13-22_00_59/last.pth"
	#"save/Contrastive/SupCon/path_models/path_dino_vit_small_p_16_SupCon_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_13-13_15_14/last.pth"
	
	#"save/Contrastive/SupCon/path_models/path_dino_vit_small_p_16_SupCon_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_31-15_17_03/last.pth"
	#"save/Contrastive/InfoNCE/path_models/path_dino_vit_small_p_16_InfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_06_01-18_12_31/last.pth"
	#"save/Contrastive/SimCLR/path_models/path_dino_vit_small_p_16_SimCLR_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_06_02-03_12_34/last.pth"
	#"save/Contrastive/SINCERE/path_models/path_dino_vit_small_p_16_SINCERE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_06_01-00_11_25/last.pth"
	"save/Contrastive/EpsSupInfoNCE/path_models/path_dino_vit_small_p_16_EpsSupInfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_06_01-09_11_32/last.pth"

)

# Parâmetros gerais do comando

NAME_DATASET="ModelNet40"
dataset="path"
dataset_train="Datasets/train_M40.csv"
dataset_test="Datasets/test_M40.csv"
n_cls=40

batch_size=128
size=224
learning_rate=0.0001
num_workers=16
model="dino_vit_small_p_16"


# Loop para executar a avaliação para cada checkpoint
for ckpt in "${checkpoints[@]}"; do
    # Nome do método extraído do caminho do checkpoint (opcional para fins de log)
    method_name=$(echo "$ckpt" | cut -d'/' -f3)

    echo "Iniciando avaliação linear para o checkpoint: $ckpt"
    time python main_linear.py \
        --batch_size $batch_size \
        --n_cls $n_cls \
        --size $size \
        --model $model \
        --dataset $dataset \
        --train_file $dataset_train \
	--test_file $dataset_test \
	--learning_rate $learning_rate \
        --num_workers $num_workers \
        --epochs 100 \
        --ckpt $ckpt \
        > "paper_experiments/LinearEval_${NAME_DATASET}/${method_name}_LinearEvalStage-${model}.txt"

    echo "Avaliação para o checkpoint $ckpt finalizada."
done
                                
