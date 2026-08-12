#!/bin/bash

#For ModelNet10
python3 inferences.py --model vit_base --n_cls 10 --dataset path --train_file Datasets/train_M10.csv --test_file Datasets/test_M10.csv --ckpt save/Linear/path_models/path_vit_base_path_bsz_128_lr_0.0001_size_224_2025_06_25-13_41_31/last.pth > infer_SINCERE_vitbase_M10.txt 

python3 inferences.py --model dino_vit_base_p_16 --n_cls 10 --dataset path --train_file Datasets/train_M10.csv --test_file Datasets/test_M10.csv --ckpt save/Linear/path_models/path_dino_vit_base_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-17_12_42/last.pth > infer_SimCLR_DINOvitbase_M10.txt


#For ModelNet40
python3 inferences.py --model dino_vit_small_p_16 --n_cls 40 --dataset path --train_file Datasets/train_M40.csv --test_file Datasets/test_M40.csv --ckpt save/Linear/path_models/path_dino_vit_small_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-20_45_23/last.pth > infer_EpsSupInfoNCE_dinovitsmall_M40.txt 

python3 inferences.py --model dino_vit_base_p_16 --n_cls 40 --dataset path --train_file Datasets/train_M40.csv --test_file Datasets/test_M40.csv --ckpt  save/Linear/path_models/path_dino_vit_base_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-23_39_57/last.pth > infer_SimCLR_dinovitbase_M40.txt


