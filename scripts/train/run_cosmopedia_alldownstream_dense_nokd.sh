#!/bin/bash

export WORLD_SIZE=32
export NODE_RANK=0
export MASTER_ADDR=192.168.201.209
export MASTER_PORT=12345

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

export SPARSITY=0

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=llama2_7b_sp${SPARSITY}_Cosmopedia_AllDownstream

export MDL=meta-llama/Llama-2-7b-hf
export MAX_SEQ_LEN=4096
export PRECISION=amp_bf16
export USE_FUSED_CROSSENTROPY_LOSS=1

export MAX_DURATION=1ep
export EVAL_INTERVAL=10000ba

# 29117 ba
export GLOBAL_BS=256
export PER_DEVICE_BS=8

export LR=1e-4
export WARMUP=1500ba

export RUN_NAME=dense_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_GradClip1_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}

composer train.py \
    yamls/pretrain/llama2_7b_cosmopedia_alldownstream_dense_nokd.yaml \
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
    precision=${PRECISION} \
    dist_timeout=10000000
