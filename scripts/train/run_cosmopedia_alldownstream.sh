#!/bin/bash

# FINETUNED TEACHER = /network/eldar/for_gsm_evals/llama2_7b_cosmopedia_alldownstream/dense_amp_bf16_maxseq4096_1ep_cosineLR1e-4_warmup1500ba_GradClip1_globalBS384_evalInterval10000ba/hf
# FOR CURRICULUM LEARNING = /home/eldar/llmfoundry_checkpoints/llama2_7b_sp70_Dolma50b/oneshot_sp70_amp_bf16_maxseq4096_1ep_cosineLR1e-4_warmup1500ba_noGradClip_globalBS192_evalInterval10000ba_KDce1.0_KDsquareh1.0/hf

export WORLD_SIZE=32
export NODE_RANK=0
export MASTER_ADDR=192.168.201.209
export MASTER_PORT=12345

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

export SPARSITY=70

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=llama2_7b_sp${SPARSITY}_Cosmopedia_AllDownstream

#export MDL=/network/eldar/llmfoundry_checkpoints/llama2_7b_cosmopedia_alldownstream/oneshot_sp70_fromConverged24_amp_bf16_maxseq4096_1ep_cosineLR1e-4_warmup1500ba_noGradClip_globalBS256_evalInterval10000ba/hf
export MDL=/network/eldar/llmfoundry_checkpoints/llama2_7b_sp70_Dolma50b/oneshot_sp70_amp_bf16_maxseq4096_1ep_cosineLR1e-4_warmup1500ba_noGradClip_globalBS192_evalInterval10000ba_KDce1.0_KDsquareh1.0

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

# Knowledge distillation
export TEACHER=meta-llama/Llama-2-7b-hf
#export TEACHER=/network/eldar/llmfoundry_checkpoints/llama2_7b_cosmopedia_alldownstream/dense_amp_bf16_maxseq4096_1ep_cosineLR1e-4_warmup1500ba_GradClip1_globalBS384_evalInterval10000ba
export KL_TEMP=0.0
export HARDNESS_KL=0.0
export HARDNESS_CE=1.0
export HARDNESS_SQUAREHEAD=1.0

export RUN_NAME=oneshot_sp${SPARSITY}_fromConverged24_fromConverged70onDolma_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_noGradClip_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}

composer train_sparse.py \
    yamls/pretrain/llama2_7b_cosmopedia_alldownstream.yaml \
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
