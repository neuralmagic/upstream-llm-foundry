#!/bin/bash

# lr = {3e-5, 5e-5}
# num_epochs = {2, 4}

export WORLD_SIZE=16
export NODE_RANK=1
export MASTER_ADDR=192.168.201.210
export MASTER_PORT=12346

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

for MAX_DURATION in 3ep;
do
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}
    export WANDB_ENTITY=eldarkurtic
    export WANDB_DISABLED=False
    export WANDB_PROJECT=llama3_70b_instruct_mathador

    export USE_FUSED_CROSSENTROPY_LOSS=1
    export MDL="meta-llama/Meta-Llama-3-70B-Instruct"

    export MAX_SEQ_LEN=512
    export PRECISION=amp_bf16

    export GLOBAL_BS=32
    export PER_DEVICE_BS=1
    export WARMUP=20ba

    export LR=3e-5

    export RUN_NAME=denseLlama3_70b_instruct_${MAX_DURATION}_lr${LR}_bs${GLOBAL_BS}_GradClip1_warmup${WARMUP}

    composer train_sparse_downstream.py \
        yamls/finetune/mathador_70b_instruct_GradClip1_cosineLR.yaml \
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
