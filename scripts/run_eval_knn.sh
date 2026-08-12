#!/bin/bash

#For ModelNet40
python eval_knn.py --arch dino_vit_small_p_16 --n_cls 40 --train_file Datasets/train_M40.csv --test_file Datasets/test_M40.csv --pretrained_weights save/Linear/path_models/path_dino_vit_small_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-20_45_23/last.pth > eval_knn_EpsSupInfoNCE_dinovitsmall_M40.txt

python eval_knn.py --arch dino_vit_base_p_16 --n_cls 40 --train_file Datasets/train_M40.csv --test_file Datasets/test_M40.csv --pretrained_weights save/Linear/path_models/path_dino_vit_base_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-23_39_57/last.pth > eval_knn_SimCLR_dinovitbase_M40.txt


#For ModelNet10
python eval_knn.py --arch vit_base --n_cls 10 --train_file Datasets/train_M10.csv --test_file Datasets/test_M10.csv --pretrained_weights save/Linear/path_models/path_vit_base_path_bsz_128_lr_0.0001_size_224_2025_06_25-13_41_31/last.pth > eval_knn_SimCLR_vitbase_M10.txt

python eval_knn.py --arch dino_vit_base_p_16 --n_cls 10 --train_file Datasets/train_M10.csv --test_file Datasets/test_M10.csv --pretrained_weights save/Linear/path_models/path_dino_vit_base_p_16_path_bsz_128_lr_0.0001_size_224_2025_06_25-17_12_42/last.pth > eval_knn_SimCLR_dinovitbase_M10.txt




