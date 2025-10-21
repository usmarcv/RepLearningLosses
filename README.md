

## Installation
```shell
 git clone this repo
```
```shell
pip install pipenv
```
```shell
pipenv shell
```
```shell
pipenv sync
```

## Running

Example bash run using `--dataset` CIFAR-10:

### (1) Pre-training stage:

```shell
time nohup pipenv run python main_supcon.py --epochs 10 --batch_size 32 --learning_rate 0.001 --temp 0.1 --cosine --method SINCERE > SINCERE_trainingPhase_Log.txt &
```

### (2) Linear evaluation stage:
```shell
insert here
```

## Downstream tasks

### Retrieval
```shell
```

### k-NN classification
```shell
```

## Acknowledgement
Our approach is built using the awesome [SupCon](https://github.com/HobbitLong/SupContrast), [SINCERE](https://github.com/tufts-ml/SupContrast), [ϵ-SupInfoNCE](https://github.com/EIDOSLAB/unbiased-contrastive-learning), and [DINOv1](https://github.com/facebookresearch/dino).

## Citation
```bib
  @inproceedings{Author2026,
  
  }
```
