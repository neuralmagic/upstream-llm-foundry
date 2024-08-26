#!/bin/bash

find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}

export DSET="alpaca_cleaned"
export MAX_SEQ_LEN=1024

export WORLD_SIZE=32
export NODE_RANK=2
export MASTER_ADDR=192.168.201.209
export MASTER_PORT=12346

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=True
export WANDB_PROJECT=debug

export MDL="/network/eldar/hf_home_on_network/meta-llama/Meta-Llama-3-70B-Instruct"

export PRECISION=amp_bf16
export USE_FUSED_CROSSENTROPY_LOSS=1

export MAX_DURATION=10ba
export EVAL_INTERVAL=1ep

export GLOBAL_BS=32
export PER_DEVICE_BS=1

export LR=1e-4
export WARMUP=0.05dur

# Knowledge distillation
# export TEACHER="meta-llama/Meta-Llama-3-8B-Instruct"
# export KL_TEMP=0.0
# export HARDNESS_KL=0.0
# export HARDNESS_CE=1.0
# export HARDNESS_SQUAREHEAD=1.0

# GradClipping = YES (maybe check if 1.0 is better than 2.0)
export RUN_NAME=debug_llama_${DSET}_masterweightsdtypebf16

# export CLEARML_PROJECT_NAME=${WANDB_PROJECT}
# export CLEARML_TASK_NAME=${RUN_NAME}

composer train_sparse_downstream.py \
    yamls/finetune/llama3_70b_test.yaml \
    master_weights_dtype="bf16" \
    dset_tag=${DSET} \
    model_name_or_path=${MDL} \
    max_seq_len=${MAX_SEQ_LEN} \
    max_duration=${MAX_DURATION} \
    eval_interval=${EVAL_INTERVAL} \
    global_train_batch_size=${GLOBAL_BS} \
    device_train_microbatch_size=${PER_DEVICE_BS} \
    device_eval_batch_size=${PER_DEVICE_BS} \
    run_name=${RUN_NAME} \
    optimizer.lr=${LR} \
    eval_first=False \
    scheduler.t_warmup=${WARMUP} \
    precision=${PRECISION} \
    dist_timeout=10000000 2>&1 | tee /home/eldar/tmp_logs/${RUN_NAME}.log

