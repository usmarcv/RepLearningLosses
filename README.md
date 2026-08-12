
## Setup

## Installation
```shell
git clone this repo
```

Build and run the provided `Dockerfile` (PyTorch 2.5.0 + CUDA 12.4 base image):

```shell
docker build -t replearninglosses .
```
```shell
docker run --gpus all -it -v $(pwd):/root replearninglosses
```

All commands below assume you're inside the container (or an equivalent environment with the dependencies installed).

## Model / method names

Valid `--model` / `--arch` values (shared across scripts):
`resnet50`, `vit_small`, `vit_base`, `dino_vit_small_p_16`, `dino_vit_small_p_8`, `dino_vit_base_p_16`, `dino_vit_base_p_8`
(`main_ce.py` also accepts `dinov2_vit_small_p_14`, `dinov2_vit_base_p_14`)

Valid `--method` values for contrastive pretraining (`main_supcon.py` only):
`SINCERE`, `SupCon`, `EpsSupInfoNCE`, `SimCLR`, `InfoNCE`

## Pipeline

Checkpoints flow between stages: `main_supcon.py` → `last.pth` feeds `--ckpt` of `main_linear.py` → its `last.pth` feeds `--pretrained_weights`/`--ckpt` of `eval_knn.py`, `eval_image_retrieval.py`, and `inferences.py`.

### 1. Build dataset CSVs

```
python transform_mv_data.py --root_dir ModelNet40/ --split train --output_csv Datasets/train_M40.csv
python transform_mv_data.py --root_dir ModelNet40/ --split test --output_csv Datasets/test_M40.csv
```

### 2. Contrastive pretraining

```
python main_supcon.py --batch_size 128 --size 224 --model resnet50 \
    --epochs 100 --learning_rate 0.0001 --dataset path \
    --train_mode contrastive-mode --train_file Datasets/train_M40.csv --num_workers 16 \
    --method SimCLR --cosine
```

Saves to `save/Contrastive/<method>/path_models/<model_name>/last.pth`.

### 3. (Alternative) Supervised cross-entropy baseline

```
python main_ce.py --batch_size 128 --learning_rate 0.0001 --epochs 100 --n_cls 40 \
    --model resnet50 --dataset path \
    --train_file Datasets/train_M40.csv --test_file Datasets/test_M40.csv \
    --size 224 --num_workers 16
```

### 4. Linear evaluation

Trains a linear classifier on a frozen checkpoint from step 2.

```
python main_linear.py --batch_size 128 --n_cls 40 --size 224 --model resnet50 \
    --dataset path --train_file Datasets/train_M40.csv --test_file Datasets/test_M40.csv \
    --learning_rate 0.0001 --num_workers 16 --epochs 100 \
    --ckpt save/Contrastive/SimCLR/path_models/<run_name>/last.pth
```

Saves to `save/Linear/path_models/<model_name>/last.pth`.

### 5. kNN evaluation

```
python eval_knn.py --arch dino_vit_small_p_16 --n_cls 40 \
    --train_file Datasets/train_M40.csv --test_file Datasets/test_M40.csv \
    --pretrained_weights save/Linear/path_models/<run_name>/last.pth
```

### 6. Image / shape retrieval evaluation

Needs `Datasets/<dataset>/gnd_<dataset>.pkl` + `Datasets/<dataset>/jpg/` present. `--dataset` is `roxford5k` or `rparis6k`.

```
python3 eval_image_retrieval.py --arch dino_vit_small_p_16 --dataset roxford5k \
    --pretrained_weights save/Linear/path_models/<run_name>/last.pth
```

### 7. Inference / accuracy report

`--ckpt` must come from step 4 (needs both `model` and `classifier` keys).

```
python3 inferences.py --model vit_base --n_cls 10 --dataset path \
    --train_file Datasets/train_M10.csv --test_file Datasets/test_M10.csv \
    --ckpt save/Linear/path_models/<run_name>/last.pth
```

### 8. t-SNE plot

No CLI args — edit `out_folders` / `class_labels` in the `__main__` block of `tsne.py` to point at your `save/Linear/...` run(s), then:

```
python tsne.py
```

## Batch / sweep scripts

`scripts/*.sh` wraps the commands above as loops/sweeps across models and losses (e.g. `run_ContrastiveMode.sh`, `run_LinearEvalMode_*.sh` via `run_LinearMaster.sh`, `run_eval_knn.sh`, `run_image_retrieval.sh`, `run_inferences.sh`). Run these from the repo root (not from inside `scripts/`) since they reference `Datasets/`, `save/`, etc. relative to the root. Update the hardcoded checkpoint paths inside a script, then run it directly:

```
bash scripts/run_ContrastiveMode.sh
```

## Acknowledgement
Our approach is built using the awesome [SupCon](https://github.com/HobbitLong/SupContrast), [SINCERE](https://github.com/tufts-ml/SupContrast), [ϵ-SupInfoNCE](https://github.com/EIDOSLAB/unbiased-contrastive-learning), and [DINOv1](https://github.com/facebookresearch/dino).

## Citation
```bib
  @inproceedings{TestingAuthor,
  
  }
```
