#!/bin/bash

# 1371 samples of 2048 seq-length

# try per-device-bs of 4 and 8

# export CUDA_VISIBLE_DEVICES=0
export DATA_LOCAL=/network/eldar/datasets/llama2_7b/openllm

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=llama2_7b_openllm

export USE_FUSED_CROSSENTROPY_LOSS=1

# /network/eldar/llama2_7b_c4
# oneshot_sparsegpt_sp40_nsamples512_seqlen2048
# oneshot_sparsegpt_sp50_nsamples512_seqlen2048
# oneshot_sparsegpt_sp60_nsamples512_seqlen2048
# oneshot_sparsegpt_sp70_nsamples512_seqlen2048

export MDL=meta-llama/Llama-2-7b-hf
export MAX_SEQ_LEN=2048

export MAX_DURATION=${1}
export EVAL_INTERVAL=1ep

export GLOBAL_BS=8
export PER_DEVICE_BS=1

export LR=3e-4
export WARMUP=10ba

export RUN_NAME=dense_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_yesGradClip1.0_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}_fusedCE${USE_FUSED_CROSSENTROPY_LOSS}

composer train.py \
    yamls/pretrain/llama2_7b_openllm.yaml \
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
    eval_first=False \
    scheduler.t_warmup=${WARMUP}
