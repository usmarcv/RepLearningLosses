#!/bin/bash

checkpoints=(
	#"save/Contrastive/EpsSupInfoNCE/path_models/path_vit_base_EpsSupInfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_12-14_08_39/last.pth"
	#"save/Contrastive/InfoNCE/path_models/path_vit_base_InfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_12-21_51_48/last.pth"
	#"save/Contrastive/SimCLR/path_models/path_vit_base_SimCLR_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_13-05_34_31/last.pth"
	"save/Contrastive/SINCERE/path_models/path_vit_base_SINCERE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_12-06_19_13/last.pth"
	#"save/Contrastive/SupCon/path_models/path_vit_base_SupCon_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_11-22_25_27/last.pth"	

	#ModelNet40
	#"save/Contrastive/SupCon/path_models/path_vit_base_SupCon_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_27-16_20_53/last.pth"
	#"save/Contrastive/SINCERE/path_models/path_vit_base_SINCERE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_28-11_24_35/last.pth"
	#"save/Contrastive/EpsSupInfoNCE/path_models/path_vit_base_EpsSupInfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_29-06_22_19/last.pth"
	#"save/Contrastive/InfoNCE/path_models/path_vit_base_InfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_30-01_20_27/last.pth"
	#"save/Contrastive/SimCLR/path_models/path_vit_base_SimCLR_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_2025_05_30-20_22_29/last.pth"

)

# Parâmetros gerais do comando
batch_size=128
n_cls=10
size=224
learning_rate=0.0001
dataset="path"
num_workers=16
model="vit_base"

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
        --train_file Datasets/train_M10.csv \
	--test_file Datasets/test_M10.csv \
	--learning_rate $learning_rate \
        --num_workers $num_workers \
        --epochs 100 \
        --ckpt $ckpt \
        > "$save_folder/LinearEval_ModelNet10/${method_name}_LinearEvalStage-${model}.txt"

    echo "Avaliação para o checkpoint $ckpt finalizada."
done
                                
