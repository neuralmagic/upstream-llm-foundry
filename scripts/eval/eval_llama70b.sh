#!/bin/bash

# export HF_DATASETS_CACHE="/localhome/ekurtic/hf_cache"
# export HF_HOME="..."

export WORLD_SIZE=32
export NODE_RANK=2
export MASTER_ADDR=192.168.201.209
export MASTER_PORT=12346

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL
export WANDB_DISABLED=True
export WANDB_ENTITY=eldarkurtic
export WANDB_PROJECT=test_evals

export BASE_MDL_PATH="meta-llama"
export USE_FUSED_CROSSENTROPY_LOSS=1
export MAX_SEQ_LEN=1024
export PER_DEVICE_BS=2
export PRECISION=amp_bf16

export PROJECT=eval_llama3_70b_instruct_${PRECISION}
export RUN_NAME=openllmNoGSM8k_seqlen${MAX_SEQ_LEN}_perdeviceBS${PER_DEVICE_BS}_worldsize${WORLD_SIZE}

export CKPT="Meta-Llama-3-70B-Instruct"
echo $CKPT
mkdir -p eval/results/${PROJECT}/${CKPT}

composer eval/eval.py \
    eval/yamls/eval_openllm_noGSM8k.yaml \
    max_seq_len=${MAX_SEQ_LEN} \
    device_eval_batch_size=${PER_DEVICE_BS} \
    dist_timeout=10000000 \
    ckpt_path=${BASE_MDL_PATH}/${CKPT} 2>&1 | tee eval/results/${PROJECT}/${CKPT}/${RUN_NAME}.txt
