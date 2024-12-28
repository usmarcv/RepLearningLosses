This part of the repo is based on:
- https://github.com/HobbitLong/SupContrast
- https://github.com/tufts-ml/SupContrast
- https://github.com/EIDOSLAB/unbiased-contrastive-learning


> git clone this repo

> pip install pipenv

> pipenv shell

> pipenv sync

> pipenv run python file.py

Example bash run using `--dataset` CIFAR-10:

>  time nohup pipenv run python main_supcon.py --epochs 10 --batch_size 32 --learning_rate 0.001 --temp 0.1 --cosine --method SINCERE > SINCERE_trainingPhase_Log.txt &
