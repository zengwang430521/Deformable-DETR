#!/usr/bin/env bash

python main.py  --output_dir ./output --enc_layers=2 --dec_layers=2 --smpl --batch_size=2


GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/smpl.sh --output_dir ./output/smpl1 --enc_layers=2 --dec_layers=2 --smpl --batch_size=2

GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/smpl.sh --output_dir ./output/smpl1 --enc_layers=2 --dec_layers=2 --smpl --batch_size=2

CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/smpl.sh --output_dir ./output/smpl1
 --enc_layers=2 --dec_layers=6 --num_queries=100 --smpl --batch_size=2 --num_workers=4 --epochs=30



python main.py  --output_dir ./output/smpl1 --enc_layers=2 --dec_layers=6 --num_queries=100 --smpl --batch_size=2 --num_workers=4 --epochs=30 --resume=output/smpl1/checkpoint.pth


# test the speed of deformable detr itself
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh --batch_size=2 --num_workers=4

# Deformable DETR 14786batch * 4gpu * 2image = 4hour


GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/smpl.sh --output_dir ./output/smpl1
--enc_layers=2 --dec_layers=6 --num_queries=100 --smpl --batch_size=2 --num_workers=4 --epochs=30
--resume=output/smpl1/checkpoint.pth
--betas_loss_coef=0.05
--pose_loss_coef=5
--mesh_loss_coef=5
--key_loss_coef=20
--key3D_loss_coef=5

GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh HA_3D deformable_detr 32 configs/smpl.sh
--output_dir ./output/smpl1
--enc_layers=2 --dec_layers=6 --num_queries=100 --smpl --batch_size=2 --num_workers=2 --epochs=30
--betas_loss_coef=0.05 --pose_loss_coef=5 --mesh_loss_coef=5 --key_loss_coef=20 --key3D_loss_coef=5
--stage=pretrain --lr=2e-4


GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh HA_3D deformable_detr 32 configs/smpl.sh
--output_dir ./output/smpl2
--enc_layers=2 --dec_layers=6 --num_queries=100 --smpl --batch_size=2 --num_workers=2 --epochs=20
--betas_loss_coef=0.05 --pose_loss_coef=5 --mesh_loss_coef=5 --key_loss_coef=20 --key3D_loss_coef=5
--stage=baseline --lr=1e-5  --pretrain=output/smp1/checkpoint.pth


GPUS_PER_NODE=1 ./tools/run_dist_slurm.sh HA_3D deformable_detr 1 configs/smpl.sh
--output_dir ./output/smpl1
--enc_layers=2 --dec_layers=6 --num_queries=100 --smpl --batch_size=2 --num_workers=1 --epochs=30
--stage=pretrain

python main.py  --output_dir ./output/debug --enc_layers=1 --dec_layers=1 --num_queries=100 --smpl --batch_size=2 --num_workers=0
