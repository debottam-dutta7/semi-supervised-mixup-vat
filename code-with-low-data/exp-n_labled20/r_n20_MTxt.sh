source /home/debottamd/.bashrc
cpython="/home/debottamd/.conda/envs/mixText/bin/python"

cd ..

CUDA_VISIBLE_DEVICES=1 ${cpython} train.py --gpu 1 --n-labeled 20 --un-labeled 500 --batch-size 2 --batch-size-u 4 --epochs 20 --val-iteration 20 \
 --lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 --lrmain 0.000005 --lrlast 0.0005 \
 --lam-v 1 --epsilon-vat 5.0
