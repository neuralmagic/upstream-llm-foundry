#!/bin/bash

export WORLD_SIZE=32
export NODE_RANK=0
export MASTER_ADDR=192.168.201.209
export MASTER_PORT=12345

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

export DATA_LOCAL=/network/eldar/datasets/llama2_7b_Cosmopedia_seq4k_tokenized

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=llama2_7b_Cosmopedia

export USE_FUSED_CROSSENTROPY_LOSS=1

# /network/eldar/llama2_7b_c4
# oneshot_sparsegpt_sp40_nsamples512_seqlen2048
# oneshot_sparsegpt_sp50_nsamples512_seqlen2048
# oneshot_sparsegpt_sp60_nsamples512_seqlen2048
# oneshot_sparsegpt_sp70_nsamples512_seqlen2048

#export MDL=meta-llama/Llama-2-7b-hf
export SPARSITY=70
export MDL=/network/eldar/llama2_7b_c4/oneshot_sparsegpt_sp${SPARSITY}_nsamples512_seqlen2048
export MAX_SEQ_LEN=4096
export PRECISION=amp_bf16

export MAX_DURATION=1ep
export EVAL_INTERVAL=5000ba

# diff in training loss: PURE vs DEFAULT
# DEFAULT + 14 per-device-bs max ~ 28k
# PURE + 16 per-device-bs max ~ 28k

export GLOBAL_BS=256
export PER_DEVICE_BS=8

export LR=3e-4
export WARMUP=1000ba

# Knowledge distillation
export TEACHER=meta-llama/Llama-2-7b-hf
export KL_TEMP=0.0
export HARDNESS_KL=0.0
export HARDNESS_CE=1.0
export HARDNESS_SQUAREHEAD=1.0

export RUN_NAME=oneshot_sp${SPARSITY}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_noGradClip_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}_fusedCE${USE_FUSED_CROSSENTROPY_LOSS}

composer train_sparse.py \
    yamls/pretrain/llama2_7b_cosmopedia.yaml \
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
    knowledge_distillation.teacher_name_or_path=${TEACHER} \
    knowledge_distillation.temperature=${KL_TEMP} \
    knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
    knowledge_distillation.hardness_ce=${HARDNESS_CE} \
    knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD} \
    dist_timeout=10000000
