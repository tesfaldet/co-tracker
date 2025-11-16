#!/bin/bash

EXP_DIR="/home/mila/m/mattie.tesfaldet/scratch/Projects/co-tracker"
EXP_NAME="cotracker3_train_kub"
DATE=`(date +'%Y-%m-%d_%H-%M-%S')`
DATASET_ROOT="/home/mila/m/mattie.tesfaldet/scratch/Projects/diffusion-pips/datasets"
NUM_STEPS=400000


echo `which python`

mkdir -p ${EXP_DIR}/${DATE}_${EXP_NAME}/logs/;
mkdir ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3;
find . \( -name "*.sh" -o -name "*.py" \) -type f -exec cp --parents {} ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3 \;

export PYTHONPATH=`(cd ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3 && pwd)`:`pwd`:$PYTHONPATH
sbatch --comment=${EXP_NAME} --partition=long  --time=3-00:00:00 --gres=gpu:l40s:4 --nodes=1 --ntasks-per-node=4 \
--job-name=${EXP_NAME} --cpus-per-task=5 --signal=USR1@60 --open-mode=append --mem-per-gpu=40GB \
--output=${EXP_DIR}/${DATE}_${EXP_NAME}/logs/%j_%x_%A_%a_%N.out \
--error=${EXP_DIR}/${DATE}_${EXP_NAME}/logs/%j_%x_%A_%a_%N.err \
--wrap=". /home/mila/m/mattie.tesfaldet/miniconda3/etc/profile.d/conda.sh; conda activate /home/mila/m/mattie.tesfaldet/miniconda3/envs/cotracker; \
srun --label python ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3/train_on_cotracker3_kubric.py --batch_size 1 \
--num_steps ${NUM_STEPS} --ckpt_path ${EXP_DIR}/${DATE}_${EXP_NAME} --model_name cotracker_three \
--save_freq 2000 --sequence_len 24 --eval_datasets tapvid_davis_first tapvid_stacking pointodyssey dynamic_replica \
--traj_per_sample 128 --sliding_window_len 16 --train_datasets cotracker3_kubric \
--save_every_n_steps 2000 --evaluate_every_n_steps 2000 --model_stride 4 --dataset_root ${DATASET_ROOT} --num_nodes 1 \
--num_virtual_tracks 64 --mixed_precision \
--corr_radius 3 --wdecay 0.0005 --linear_layer_for_vis_conf --validate_at_start --add_huber_loss"
