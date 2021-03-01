#!/usr/bin/env bash

python main.py  --output_dir ./output --enc_layers=2 --dec_layers=2 --smpl --batch_size=2


GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/smpl.sh --output_dir ./output/smpl1 --enc_layers=2 --dec_layers=2 --smpl --batch_size=2

GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/smpl.sh --output_dir ./output/smpl1 --enc_layers=2 --dec_layers=2 --smpl --batch_size=2