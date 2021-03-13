#!/usr/bin/env bash

python main.py  --output_dir ./output --enc_layers=2 --dec_layers=2 --smpl --batch_size=2


GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/smpl.sh --output_dir ./output/smpl1 --enc_layers=2 --dec_layers=2 --smpl --batch_size=2

GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/smpl.sh --output_dir ./output/smpl1 --enc_layers=2 --dec_layers=2 --smpl --batch_size=2

CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/smpl.sh --output_dir ./output/smpl1
 --enc_layers=2 --dec_layers=6 --num_queries=100 --smpl --batch_size=2 --num_workers=4 --epochs=30

GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/smpl.sh --output_dir ./output/smpl1
 --enc_layers=2 --dec_layers=6 --num_queries=100 --smpl --batch_size=2 --num_workers=4 --epochs=30
 --resume=output/smpl1/checkpoint.pth

python main.py  --output_dir ./output/smpl1 --enc_layers=2 --dec_layers=6 --num_queries=100 --smpl --batch_size=2 --num_workers=4 --epochs=30 --resume=output/smpl1/checkpoint.pth

# test the speed of deformable detr itself
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh --batch_size=2 --num_workers=4

# Deformable DETR 14786batch * 4gpu * 2image = 4hour