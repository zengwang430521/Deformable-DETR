#!/usr/bin/env bash

python main.py  --output_dir ./output --num_workers=4 --enc_layers=2 --dec_layers=2 --smpl


GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/smpl.sh --output_dir ./output --num_workers=4 --enc_layers=4 --dec_layers=4 --smpl --batch_size=8