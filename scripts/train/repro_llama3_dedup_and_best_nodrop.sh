#!/bin/bash

# ============= small dset, no dropping, wEOS ===========================
find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}

export WORLD_SIZE=32
export NODE_RANK=0
export MASTER_ADDR=192.168.201.209
export MASTER_PORT=12346

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

export SPARSITY=24

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=llama3_8b_sp${SPARSITY}_Cosmopedia_AllDownstream
export CLEARML_PROJECT_NAME=${WANDB_PROJECT}

export MDL=/network/eldar/llama3_8b_base/c4/oneshot_sparsegpt_sp2_4_nsamples512_seqlen2048_wEOS
export USE_FUSED_CROSSENTROPY_LOSS=1

export MAX_SEQ_LEN=8192
export MAX_DURATION=1ep
export EVAL_INTERVAL=1500ba
export PRECISION=amp_bf16

export GLOBAL_BS=128
export PER_DEVICE_BS=2

export LR=1e-4
export WARMUP=150ba

# Knowledge distillation
export TEACHER=meta-llama/Meta-Llama-3-8B
export KL_TEMP=0.0
export HARDNESS_KL=0.0
export HARDNESS_CE=1.0
export HARDNESS_SQUAREHEAD=1.0

export RUN_NAME=wPositionIDS_noDroppingSeqPack_DownFixed_ShubhraDedup_oneshot_wEOS_sp${SPARSITY}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_GradClip2_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}_wEOS_noWrap
export CLEARML_TASK_NAME=${RUN_NAME}

composer train_sparse_with_kd.py \
    yamls/pretrain/llama3_8b_alldownstreamfixed_shubhradedup_GradClip2.yaml \
    variables.model_name_or_path=${MDL} \
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
    scheduler.t_warmup=${WARMUP} \
    knowledge_distillation.teacher_name_or_path=${TEACHER} \
    knowledge_distillation.temperature=${KL_TEMP} \
    knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
    knowledge_distillation.hardness_ce=${HARDNESS_CE} \
    knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD}


# ============= small dset, no dropping, wBOSandEOS ===========================
find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}

export WORLD_SIZE=32
export NODE_RANK=0
export MASTER_ADDR=192.168.201.209
export MASTER_PORT=12346

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

export SPARSITY=24

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=llama3_8b_sp${SPARSITY}_Cosmopedia_AllDownstream
export CLEARML_PROJECT_NAME=${WANDB_PROJECT}

export MDL=/network/eldar/llama3_8b_base/c4/oneshot_sparsegpt_sp2_4_nsamples512_seqlen2048_wEOS
export USE_FUSED_CROSSENTROPY_LOSS=1

export MAX_SEQ_LEN=8192
export MAX_DURATION=1ep
export EVAL_INTERVAL=1500ba
export PRECISION=amp_bf16

export GLOBAL_BS=128
export PER_DEVICE_BS=2

export LR=1e-4
export WARMUP=150ba

# Knowledge distillation
export TEACHER=meta-llama/Meta-Llama-3-8B
export KL_TEMP=0.0
export HARDNESS_KL=0.0
export HARDNESS_CE=1.0
export HARDNESS_SQUAREHEAD=1.0

export RUN_NAME=wPositionIDS_wBOSandEOS_noDroppingSeqPack_DownFixed_ShubhraDedup_oneshot_wEOS_sp${SPARSITY}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_GradClip2_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}_wEOS_noWrap
export CLEARML_TASK_NAME=${RUN_NAME}

composer train_sparse_with_kd.py \
    yamls/pretrain/llama3_8b_alldownstreamfixed_shubhradedup_GradClip2_wBOSandEOS.yaml \
    variables.model_name_or_path=${MDL} \
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
    scheduler.t_warmup=${WARMUP} \
    knowledge_distillation.teacher_name_or_path=${TEACHER} \
    knowledge_distillation.temperature=${KL_TEMP} \
    knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
    knowledge_distillation.hardness_ce=${HARDNESS_CE} \
    knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD}

# ============= large dset, no dropping, wEOS ===========================
find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}

export WORLD_SIZE=32
export NODE_RANK=0
export MASTER_ADDR=192.168.201.209
export MASTER_PORT=12346

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

export SPARSITY=24

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=llama3_8b_sp${SPARSITY}_Cosmopedia_AllDownstream
export CLEARML_PROJECT_NAME=${WANDB_PROJECT}

export MDL=/network/eldar/llama3_8b_base/c4/oneshot_sparsegpt_sp2_4_nsamples512_seqlen2048_wEOS
export USE_FUSED_CROSSENTROPY_LOSS=1

export MAX_SEQ_LEN=8192
export MAX_DURATION=1ep
export EVAL_INTERVAL=6000ba
export PRECISION=amp_bf16

export GLOBAL_BS=128
export PER_DEVICE_BS=2

export LR=1e-4
export WARMUP=600ba

# Knowledge distillation
export TEACHER=meta-llama/Meta-Llama-3-8B
export KL_TEMP=0.0
export HARDNESS_KL=0.0
export HARDNESS_CE=1.0
export HARDNESS_SQUAREHEAD=1.0

export RUN_NAME=wPositionIDS_noDroppingSeqPack_DownFixedShubhraDedup_SYN8BwdiffsOpenLLM4x_SYN8Bopenmath1x_FineWebEdu10BT_oneshot_wEOS_sp${SPARSITY}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_GradClip2_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}_wEOS_noWrap
export CLEARML_TASK_NAME=${RUN_NAME}

composer train_sparse_with_kd.py \
    yamls/pretrain/llama3_8b_alldownstreamfixed_shubhradedup_SYN8BwdiffsOpenLLM4x_SYN8Bopenmath1x_FineWebEdu10BT_GradClip2.yaml \
    variables.model_name_or_path=${MDL} \
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
    scheduler.t_warmup=${WARMUP} \
    knowledge_distillation.teacher_name_or_path=${TEACHER} \
    knowledge_distillation.temperature=${KL_TEMP} \
    knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
    knowledge_distillation.hardness_ce=${HARDNESS_CE} \
    knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD}

# ============= large dset, no dropping, wBOSandEOS ===========================
find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}

export WORLD_SIZE=32
export NODE_RANK=0
export MASTER_ADDR=192.168.201.209
export MASTER_PORT=12346

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

export SPARSITY=24

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=llama3_8b_sp${SPARSITY}_Cosmopedia_AllDownstream
export CLEARML_PROJECT_NAME=${WANDB_PROJECT}

export MDL=/network/eldar/llama3_8b_base/c4/oneshot_sparsegpt_sp2_4_nsamples512_seqlen2048_wEOS
export USE_FUSED_CROSSENTROPY_LOSS=1

export MAX_SEQ_LEN=8192
export MAX_DURATION=1ep
export EVAL_INTERVAL=6000ba
export PRECISION=amp_bf16

export GLOBAL_BS=128
export PER_DEVICE_BS=2

export LR=1e-4
export WARMUP=600ba

# Knowledge distillation
export TEACHER=meta-llama/Meta-Llama-3-8B
export KL_TEMP=0.0
export HARDNESS_KL=0.0
export HARDNESS_CE=1.0
export HARDNESS_SQUAREHEAD=1.0

export RUN_NAME=wPositionIDS_wBOSandEOS_noDroppingSeqPack_DownFixedShubhraDedup_SYN8BwdiffsOpenLLM4x_SYN8Bopenmath1x_FineWebEdu10BT_oneshot_wEOS_sp${SPARSITY}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_GradClip2_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}_wEOS_noWrap
export CLEARML_TASK_NAME=${RUN_NAME}

composer train_sparse_with_kd.py \
    yamls/pretrain/llama3_8b_alldownstreamfixed_shubhradedup_SYN8BwdiffsOpenLLM4x_SYN8Bopenmath1x_FineWebEdu10BT_GradClip2_wBOSandEOS.yaml \
    variables.model_name_or_path=${MDL} \
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
    scheduler.t_warmup=${WARMUP} \
    knowledge_distillation.teacher_name_or_path=${TEACHER} \
    knowledge_distillation.temperature=${KL_TEMP} \
    knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
    knowledge_distillation.hardness_ce=${HARDNESS_CE} \
    knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD}
