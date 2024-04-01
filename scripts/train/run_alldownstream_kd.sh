#!/bin/bash

export WORLD_SIZE=24
export NODE_RANK=0
export MASTER_ADDR=192.168.201.158
export MASTER_PORT=12345

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

export SPARSITY=70
export MDL_TAG=oneshot_sparsegpt_sp${SPARSITY}_nsamples512_seqlen2048  # TODO: for checkpoints only
export DSET_TAG=dset_concats  # TODO: for checkpoints only

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=llama2_7b_sp${SPARSITY}_${DSET_TAG}

export MDL=/network/eldar/llama2_7b_c4/${MDL_TAG}
export MAX_SEQ_LEN=4096
export PRECISION=amp_bf16
export USE_FUSED_CROSSENTROPY_LOSS=1

# 7026733056 / (4096 x 192) = 8934ba
export MAX_DURATION=1ep
export EVAL_INTERVAL=2500ba

export GLOBAL_BS=192
export PER_DEVICE_BS=8

export LR=1e-4
export WARMUP=400ba

# Knowledge distillation
export TEACHER=meta-llama/Llama-2-7b-hf
export KL_TEMP=0.0
export HARDNESS_KL=0.0
export HARDNESS_CE=1.0
export HARDNESS_SQUAREHEAD=1.0

export RUN_NAME=openplatypus+openorca+openhermes25+ultrachat+dolphin+mmluaux+alpacaclean+dollyhhrlhf+flan2021+flancot+winograndeall+arcall+hellaswagall_${MDL_TAG}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_noGradClip_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}_fusedCE${USE_FUSED_CROSSENTROPY_LOSS}_wKD_CE${HARDNESS_CE}_SQ${HARDNESS_SQUAREHEAD}

composer train_sparse.py \
    yamls/pretrain/llama2_7b_dset_ablations_concats.yaml \
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
    model_tag=llama2_7b_${MDL_TAG} \
    dset_tag=${DSET_TAG} \
    knowledge_distillation.teacher_name_or_path=${TEACHER} \
    knowledge_distillation.temperature=${KL_TEMP} \
    knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
    knowledge_distillation.hardness_ce=${HARDNESS_CE} \
    knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD} \
    dist_timeout=10000000
