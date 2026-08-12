#!/bin/bash

checkpoints=(
	#"save/Contrastive/EpsSupInfoNCE/path_models/path_vit_small_EpsSupInfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_08-19_54_22/last.pth"
	#"save/Contrastive/InfoNCE/path_models/path_vit_small_InfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_08-23_34_44/last.pth"
	#"save/Contrastive/SimCLR/path_models/path_vit_small_SimCLR_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_09-03_14_38/last.pth"
	#"save/Contrastive/SupCon/path_models/path_vit_small_SupCon_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_16-14_05_21/last.pth"
	#"save/Contrastive/SINCERE/path_models/path_vit_small_SINCERE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_08-16_14_19/last.pth"
	#"save/Contrastive/SupCon/path_models/path_vit_small_SupCon_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_22-11_50_53/last.pth"

	"save/Contrastive/SupCon/path_models/path_vit_small_SupCon_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_23-04_54_50/last.pth"
	"save/Contrastive/SINCERE/path_models/path_vit_small_SINCERE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_23-13_48_26/last.pth"
	"save/Contrastive/SimCLR/path_models/path_vit_small_SimCLR_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_24-16_46_23/last.pth"
	"save/Contrastive/InfoNCE/path_models/path_vit_small_InfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_24-07_47_10/last.pth"
	"save/Contrastive/EpsSupInfoNCE/path_models/path_vit_small_EpsSupInfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_23-22_47_55/last.pth"
)

# Parâmetros gerais do comando
batch_size=128
n_cls=40
size=224
learning_rate=0.0001
dataset="path"
num_workers=16
model="vit_small"

save_folder="paper_experiments"
mkdir -p "$save_folder"

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
        --train_file Datasets/train_M40.csv \
	--test_file Datasets/test_M40.csv \
	--learning_rate $learning_rate \
        --num_workers $num_workers \
        --epochs 100 \
        --ckpt $ckpt \
        > "$save_folder/LinearEval_ModelNet40/${method_name}_LinearEvalStage-${model}.txt"

    echo "Avaliação para o checkpoint $ckpt finalizada."
done
                                
