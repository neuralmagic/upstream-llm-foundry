#!/bin/bash

find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}

export SPARSITY=24

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=True
export WANDB_PROJECT=llama3_8b_sp${SPARSITY}_Cosmopedia_AllDownstream

export MDL=/network/eldar/llama3_8b_base/c4/oneshot_sparsegpt_sp2_4_nsamples512_seqlen2048_wEOS

export MAX_SEQ_LEN=8192
export PRECISION=amp_bf16
export USE_FUSED_CROSSENTROPY_LOSS=1

export MAX_DURATION=1ep
export EVAL_INTERVAL=3000ba

# 12719 ba
export GLOBAL_BS=2
export PER_DEVICE_BS=2

export LR=1e-4
export WARMUP=300ba

# Knowledge distillation
export TEACHER=meta-llama/Meta-Llama-3-8B
export KL_TEMP=0.0
export HARDNESS_KL=0.0
export HARDNESS_CE=1.0
export HARDNESS_SQUAREHEAD=1.0

# GradClipping = YES (maybe check if 1.0 is better than 2.0)
# Wrapping = NO
export RUN_NAME=test

composer train_sparse.py \
    yamls/pretrain/llama3_8b_alldownstreamfixed_cosmosubsets_automathkhanstaxstanford_finewebedu10bt_GradClip2.yaml \
    model_name_or_path=${MDL} \
    max_seq_len=${MAX_SEQ_LEN} \
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
