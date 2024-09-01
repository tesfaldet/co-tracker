#!/bin/bash

EXP_DIR="./experiments"
EXP_NAME=$1
DATE=`(date +'%Y-%m-%d_%H-%M-%S')`


echo `which python`

mkdir -p ${EXP_DIR}/${DATE}_${EXP_NAME}/logs/;

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
sbatch --comment=${EXP_NAME} --partition=long --time=2-00:00:00 --gres=gpu:rtx8000:4 --nodes=1 --ntasks-per-node=4 \
--job-name=${EXP_NAME} --cpus-per-task=6 --signal=USR1@120 --open-mode=append --mem-per-gpu=18GB \
--output=${EXP_DIR}/${DATE}_${EXP_NAME}/logs/%j_%x_%A_%a_%N.out \
--error=${EXP_DIR}/${DATE}_${EXP_NAME}/logs/%j_%x_%A_%a_%N.err \
--wrap="module load miniconda/3; conda activate /home/mila/m/mattie.tesfaldet/miniconda3/envs/cotracker; \
srun --label python ./train.py --batch_size 2 \
--num_steps 200000 --ckpt_path ${EXP_DIR}/${DATE}_${EXP_NAME} --model_name cotracker \
--save_freq 200 --sequence_len 24 --eval_datasets tapvid_davis_first \
--traj_per_sample 128 --sliding_window_len 16 \
--save_every_n_epoch 1 --evaluate_every_n_epoch 10 --model_stride 4 --num_nodes 1 \
--num_virtual_tracks 64"
