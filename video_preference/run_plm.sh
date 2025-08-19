#!/bin/bash
# baseline
# python3 run_plm.py --adapt --his-window 3 --fut-window 3 --plm-type llama --plm-size base --epochs 50 --bs 1 --lr 1e-4 --device cuda:0 --steps-per-valid 500 --save-checkpoint-per-epoch 1 --rank 32 --scheduled-sampling --video-len 10 --loss-func bce
python3 run_plm.py --adapt --his-window 3 --fut-window 3 --plm-type llama --plm-size base --epochs 60 --bs 1 --lr 1e-4 --device cuda:0 --steps-per-valid 500 --save-checkpoint-per-epoch 20 --rank 32 --scheduled-sampling --video-len 10 --loss-func bce
