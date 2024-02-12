#!/bin/bash

# 2482176 / 2048 / 256 = 4.73 steps

export CUDA_VISIBLE_DEVICES=${1}
export DATA_LOCAL=/network/eldar/datasets/opt125m/openllm

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=opt125m_openllm

export USE_FUSED_CROSSENTROPY_LOSS=1

export MDL=facebook/opt-125m
export MAX_SEQ_LEN=2048
export PRECISION=${3}

# 10b tokens --> 10*10^9/(2048*256) = 19073ba
export MAX_DURATION=${2}
export EVAL_INTERVAL=1ep

# 2048 * 256 = 524_288
export GLOBAL_BS=32
export PER_DEVICE_BS=32

export LR=3e-4
export WARMUP=10ba

export RUN_NAME=dense_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_yesGradClip1.0_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}_fusedCE${USE_FUSED_CROSSENTROPY_LOSS}

composer train.py \
    yamls/pretrain/opt125m_openllm.yaml \
    model_name_or_path=${MDL} \
    max_seq_len=${MAX_SEQ_LEN} \
    data_local=${DATA_LOCAL} \
    max_duration=${MAX_DURATION} \
    eval_interval=${EVAL_INTERVAL} \
    global_train_batch_size=${GLOBAL_BS} \
    device_train_microbatch_size=${PER_DEVICE_BS} \
    device_eval_batch_size=${PER_DEVICE_BS} \
    run_name=${RUN_NAME} \
    optimizer.lr=${LR} \
    eval_first=True \
    scheduler.t_warmup=${WARMUP} \
    precision=${PRECISION}
