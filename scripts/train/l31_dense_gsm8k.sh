#!/bin/bash

# ==== finetune a dense model on GSM8k ====

# Hyperparameter sweep:
# 1) lr = {1e-4, 5e-5, 3e-5}
# 2) num_epochs = {1, 2}

export CUDA_VISIBLE_DEVICES=0,1,2,3
export SPARSITY=0

export DATASET_NAME="gsm8k"
export DATASET_PATH="/network/eldar/datasets/downstream/llama31_8b/gsm8k_seqlen512"

for LR in 1e-4 5e-5;
do
    for MAX_DURATION in 1ep;
    do
        find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}
        python -c "from streaming.base.util import clean_stale_shared_memory;clean_stale_shared_memory()"

        export WANDB_ENTITY=eldarkurtic
        export WANDB_DISABLED=False
        export WANDB_PROJECT=llama31_8b_sp${SPARSITY}_${DATASET_NAME}
        export CLEARML_PROJECT_NAME=${WANDB_PROJECT}

        export MDL=meta-llama/Llama-3.1-8B
        export USE_FUSED_CROSSENTROPY_LOSS=1

        export MAX_SEQ_LEN=512
        export EVAL_INTERVAL=1ep
        export PRECISION=amp_bf16

        export GLOBAL_BS=32
        export PER_DEVICE_BS=8
        export WARMUP=20ba

        # export TEACHER=meta-llama/Llama-3.1-8B
        # export KL_TEMP=0.0
        # export HARDNESS_KL=0.0
        # export HARDNESS_CE=1.0
        # export HARDNESS_SQUAREHEAD=1.0

        export RUN_NAME=sp${SPARSITY}_${MAX_DURATION}_lr${LR}_bs${GLOBAL_BS}_GradClip2_warmup${WARMUP}
        export CLEARML_TASK_NAME=${RUN_NAME}

        composer train_sparse_with_kd.py \
            yamls/finetune/llama31_8b_GradClip2_gsm8k.yaml \
            variables.model_name_or_path=${MDL} \
            variables.project_name=${WANDB_PROJECT} \
            variables.dataset_path=${DATASET_PATH} \
            run_name=${RUN_NAME} \
            max_seq_len=${MAX_SEQ_LEN} \
            precision=${PRECISION} \
            max_duration=${MAX_DURATION} \
            eval_interval=${EVAL_INTERVAL} \
            eval_first=True \
            global_train_batch_size=${GLOBAL_BS} \
            device_train_microbatch_size=${PER_DEVICE_BS} \
            device_eval_batch_size=${PER_DEVICE_BS} \
            dist_timeout=10000000 \
            optimizer.lr=${LR} \
            scheduler.t_warmup=${WARMUP}
            # knowledge_distillation.teacher_name_or_path=${TEACHER} \
            # knowledge_distillation.temperature=${KL_TEMP} \
            # knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
            # knowledge_distillation.hardness_ce=${HARDNESS_CE} \
            # knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD}

    done
done