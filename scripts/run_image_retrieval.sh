#!/bin/bash


#For Modelnet40 eval
python3 eval_image_retrieval.py --arch dino_vit_small_p_16 --dataset roxford5k --pretrained_weights save/Linear/path_models/path_dino_vit_small_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-20_45_23/last.pth > roxford5k_EpsSupInfoNCE_dinovitsmall_M40.txt

python3 eval_image_retrieval.py --arch dino_vit_small_p_16 --dataset rparis6k --pretrained_weights save/Linear/path_models/path_dino_vit_small_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-20_45_23/last.pth > rparis6k_EpsSupInfoNCE_dinovitsmall_M40.txt

python3 eval_image_retrieval.py --arch dino_vit_base_p_16 --dataset roxford5k --pretrained_weights save/Linear/path_models/path_dino_vit_base_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-23_39_57/last.pth > roxford5k_SimCLR_dinovitbase_M40.txt

python3 eval_image_retrieval.py --arch dino_vit_base_p_16 --dataset rparis6k --pretrained_weights save/Linear/path_models/path_dino_vit_base_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-23_39_57/last.pth > rparis6k_SimCLR_dinovitbase_M40.txt


#For ModelNet10 eval
python3 eval_image_retrieval.py --arch vit_base --dataset roxford5k --pretrained_weights save/Linear/path_models/path_vit_base_path_bsz_128_lr_0.0001_size_224_2025_06_25-13_41_31/last.pth > roxford5k_SINCERE_vitbase_M10.txt

python3 eval_image_retrieval.py --arch vit_base --dataset rparis6k --pretrained_weights save/Linear/path_models/path_vit_base_path_bsz_128_lr_0.0001_size_224_2025_06_25-13_41_31/last.pth > rparis6k_SINCERE_vitbase_M10.txt

python3 eval_image_retrieval.py --arch dino_vit_base_p_16 --dataset roxford5k --pretrained_weights save/Linear/path_models/path_dino_vit_base_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-17_12_42/last.pth > roxford5k_SimCLR_dinovitbase_M10.txt

python3 eval_image_retrieval.py --arch dino_vit_base_p_16 --dataset rparis6k --pretrained_weights save/Linear/path_models/path_dino_vit_base_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-17_12_42/last.pth > rparis6k_SimCLR_dinovitbase_M10.txt
