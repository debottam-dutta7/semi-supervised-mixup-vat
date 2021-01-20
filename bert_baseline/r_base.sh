source /home/debottamd/.bashrc
cpython="/home/debottamd/.conda/envs/mixText/bin/python"


CUDA_VISIBLE_DEVICES=1 ${cpython} normal_train.py --gpu 1 --n-labeled 200 --batch-size 8 --epochs 20 
