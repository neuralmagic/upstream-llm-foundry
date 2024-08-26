#!/bin/bash

# lr = {3e-5, 5e-5}
# num_epochs = {2, 4}

for MAX_DURATION in 2ep 4ep;
do
    export CUDA_VISIBLE_DEVICES=4,5,6,7

    find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}
    export WANDB_ENTITY=eldarkurtic
    export WANDB_DISABLED=False
    export WANDB_PROJECT=llama2_mathador

    export USE_FUSED_CROSSENTROPY_LOSS=1
    export MDL="meta-llama/Llama-2-7b-hf"

    export MAX_SEQ_LEN=512
    export PRECISION=amp_bf16

    export GLOBAL_BS=32
    export PER_DEVICE_BS=8
    export WARMUP=20ba

    export LR=5e-5

    export RUN_NAME=denseLlama2_7b_${MAX_DURATION}_lr${LR}_bs${GLOBAL_BS}_GradClip1_warmup${WARMUP}

    composer train_sparse_downstream.py \
        yamls/finetune/mathador_GradClip1_cosineLR.yaml \
        model_name_or_path=${MDL} \
        max_seq_len=${MAX_SEQ_LEN} \
        max_duration=${MAX_DURATION} \
        global_train_batch_size=${GLOBAL_BS} \
        device_train_microbatch_size=${PER_DEVICE_BS} \
        device_eval_batch_size=${PER_DEVICE_BS} \
        run_name=${RUN_NAME} \
        optimizer.lr=${LR} \
        scheduler.t_warmup=${WARMUP} \
        precision=${PRECISION}
done
