#!/usr/bin/env bash

python eval.py --enc_layers=2 --dec_layers=6 --num_queries=100 --smpl --eval --eval_dataset=full_h36m --resume=output/smpl1/checkpoint.pth