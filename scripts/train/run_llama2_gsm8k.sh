#!/bin/bash

# sparsity = {70}
# lr = {1e-4, 8e-5, 5e-5, 3e-5, 1e-5}
# LOOP -> num_epochs = {2, 3, 4, 5}

#/network/eldar/cerebras/Llama-2-7b-pruned70-retrained

export CUDA_VISIBLE_DEVICES=0,1,2,3

export SPARSITY=24
export LR=3e-5

# 2ep 3ep 4ep 5ep;
for MAX_DURATION in 1ep;
do

    find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}
    export WANDB_ENTITY=eldarkurtic
    export WANDB_DISABLED=False
    export WANDB_PROJECT=llama2_gsm8k_sp${SPARSITY}_${MAX_DURATION}

    export USE_FUSED_CROSSENTROPY_LOSS=1
    export MDL=/network/eldar/models_to_share/llama2_7b_sp${SPARSITY}_v1

    export MAX_SEQ_LEN=512
    export PRECISION=amp_bf16
    export EVAL_INTERVAL=1ep

    export GLOBAL_BS=32
    export PER_DEVICE_BS=8

    export WARMUP=20ba

    # Knowledge distillation
    export TEACHER=/network/eldar/llama2_7b_gsm8k/dense
    export KL_TEMP=0.0
    export HARDNESS_KL=0.0
    export HARDNESS_CE=1.0
    export HARDNESS_SQUAREHEAD=1.0

    export RUN_NAME=NMmdl_CE${HARDNESS_CE}_SquareHead${HARDNESS_SQUAREHEAD}_oneshot_sp${SPARSITY}_uniform_${MAX_DURATION}_lr${LR}_bs${GLOBAL_BS}_noGradClip_warmup${WARMUP}

    composer train_sparse_downstream.py \
        yamls/finetune/FT_gsm8k_noGradClip_KD_linearLR.yaml \
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
        knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD}
done