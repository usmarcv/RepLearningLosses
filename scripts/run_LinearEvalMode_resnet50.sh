#!/bin/bash

checkpoints=(
  # "save/Contrastive/EpsSupInfoNCE/path_models/path_resnet50_EpsSupInfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_warm_2025_05_10-08_17_57/last.pth" 
  # "save/Contrastive/InfoNCE/path_models/path_resnet50_InfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_warm_2025_05_10-21_01_10/last.pth"
  # "save/Contrastive/SimCLR/path_models/path_resnet50_SimCLR_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_warm_2025_05_11-09_44_31/last.pth"
  # "save/Contrastive/SINCERE/path_models/path_resnet50_SINCERE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_warm_2025_05_09-19_33_54/last.pth"
  # "save/Contrastive/SupCon/path_models/path_resnet50_SupCon_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_warm_2025_05_09-06_52_22/last.pth"
	
#  "save/Contrastive/SupCon/path_models/path_resnet50_SupCon_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_warm_2025_05_25-01_40_03/last.pth"
 # "save/Contrastive/SINCERE/path_models/path_resnet50_SINCERE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_warm_2025_05_26-08_54_52/last.pth"
 # "save/Contrastive/EpsSupInfoNCE/path_models/path_resnet50_EpsSupInfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_warm_2025_06_06-15_28_00/last.pth"
 # "save/Contrastive/InfoNCE/path_models/path_resnet50_InfoNCE_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_warm_2025_06_07-23_46_42/last.pth"
 # "save/Contrastive/SimCLR/path_models/path_resnet50_SimCLR_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_warm_2025_06_09-12_04_23/last.pth"
 "save/Contrastive/SimCLR/path_models/path_resnet50_SimCLR_bsz_128_lr_0.0001_size_224_temp_0.07_wdecay_0.0001_trial_0_cosine_warm_2025_06_23-10_37_21/last.pth"
	 
)

# Parâmetros gerais do comando
batch_size=128
n_cls=40
size=224
learning_rate=0.0001
dataset="path"
num_workers=16
model="resnet50"

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
                                
