#!/bin/bash

# "garage-bAInd/Open-Platypus 3870
# "Open-Orca/OpenOrca 8170
# "cognitivecomputations/dolphin flan1m 8170
# "cognitivecomputations/dolphin flam5m 8168
# "teknium/OpenHermes-2.5 4036
# "stingning/ultrachat 4117
# "yahma/alpaca-cleaned 1035
# "mosaicml/dolly_hhrlhf 5813
# "chiayewken/flan-cot 8075

# tuples=("open_platypus 4096" "alpaca_cleaned 1024" "dolly_hhrlhf 4096" "dolphin_flan1m 8192" "dolphin_flan5m 8192" "open_orca 8192" "open_hermes_2_5 4096" "ultrachat 4096" "flan_cot 8192")

export DSET="open_platypus"
export MAX_SEQ_LEN=4096

find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}
# Split the tuple into two parts
# export DSET=$(echo $tuple | cut -d ' ' -f 1)
# export MAX_SEQ_LEN=$(echo $tuple | cut -d ' ' -f 2)

export WORLD_SIZE=32
export NODE_RANK=2
export MASTER_ADDR=192.168.201.209
export MASTER_PORT=12346

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

export SPARSITY=24

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=llama3_8b_instruct_sp${SPARSITY}

export MDL="/network/eldar/llama3_8b_instruct/open_platypus/SP2of4_DSETplatypus_train_NSAMPLES1024_PERCDAMP0.05_ISCHATFORMATTrue"

# export MAX_SEQ_LEN=8192  SET ABOVE FROM TUPLE !
export PRECISION=amp_bf16
export USE_FUSED_CROSSENTROPY_LOSS=1

export MAX_DURATION=10ba
export EVAL_INTERVAL=1ep

export GLOBAL_BS=128
export PER_DEVICE_BS=2

export LR=1e-4
export WARMUP=50ba

# Knowledge distillation
export TEACHER="meta-llama/Meta-Llama-3-8B-Instruct"
export KL_TEMP=0.0
export HARDNESS_KL=0.0
export HARDNESS_CE=1.0
export HARDNESS_SQUAREHEAD=1.0

# GradClipping = YES (maybe check if 1.0 is better than 2.0)
export RUN_NAME=${DSET}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_GradClip2_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}

export CLEARML_PROJECT_NAME=eldar_${WANDB_PROJECT}
export CLEARML_TASK_NAME=${RUN_NAME}

composer train_sparse_downstream.py \
    yamls/finetune/llama3_8b_instruct_dset_sweep_GradClip2.yaml \
    dset_tag=${DSET} \
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
    knowledge_distillation.teacher_name_or_path=${TEACHER} \
    knowledge_distillation.temperature=${KL_TEMP} \
    knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
    knowledge_distillation.hardness_ce=${HARDNESS_CE} \
    knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD} \
    dist_timeout=10000000 2>&1 | tee -a /home/eldar/tmp_logs/${RUN_NAME}.log
# # Loop through the array
# for tuple in "${tuples[@]}"
# do

pkill -f -9 upstream
pkill -f -9 wandb
export DSET="alpaca_cleaned"
export MAX_SEQ_LEN=1024

find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}
# Split the tuple into two parts
# export DSET=$(echo $tuple | cut -d ' ' -f 1)
# export MAX_SEQ_LEN=$(echo $tuple | cut -d ' ' -f 2)

export WORLD_SIZE=32
export NODE_RANK=2
export MASTER_ADDR=192.168.201.209
export MASTER_PORT=12346

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_CUDA_SUPPORT=1
export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

export SPARSITY=24

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=llama3_8b_instruct_sp${SPARSITY}

export MDL="/network/eldar/llama3_8b_instruct/open_platypus/SP2of4_DSETplatypus_train_NSAMPLES1024_PERCDAMP0.05_ISCHATFORMATTrue"

# export MAX_SEQ_LEN=8192  SET ABOVE FROM TUPLE !
export PRECISION=amp_bf16
export USE_FUSED_CROSSENTROPY_LOSS=1

export MAX_DURATION=10ba
export EVAL_INTERVAL=1ep

export GLOBAL_BS=128
export PER_DEVICE_BS=2

export LR=1e-4
export WARMUP=50ba

# Knowledge distillation
export TEACHER="meta-llama/Meta-Llama-3-8B-Instruct"
export KL_TEMP=0.0
export HARDNESS_KL=0.0
export HARDNESS_CE=1.0
export HARDNESS_SQUAREHEAD=1.0

# GradClipping = YES (maybe check if 1.0 is better than 2.0)
export RUN_NAME=${DSET}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_GradClip2_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}

export CLEARML_PROJECT_NAME=eldar_${WANDB_PROJECT}
export CLEARML_TASK_NAME=${RUN_NAME}

composer train_sparse_downstream.py \
    yamls/finetune/llama3_8b_instruct_dset_sweep_GradClip2.yaml \
    dset_tag=${DSET} \
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
    knowledge_distillation.teacher_name_or_path=${TEACHER} \
    knowledge_distillation.temperature=${KL_TEMP} \
    knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
    knowledge_distillation.hardness_ce=${HARDNESS_CE} \
    knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD} \
    dist_timeout=10000000 2>&1 | tee -a /home/eldar/tmp_logs/${RUN_NAME}.log

# export DSET="dolphin_flan1m"
# export MAX_SEQ_LEN=8192

# find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}
# # Split the tuple into two parts
# # export DSET=$(echo $tuple | cut -d ' ' -f 1)
# # export MAX_SEQ_LEN=$(echo $tuple | cut -d ' ' -f 2)

# export WORLD_SIZE=32
# export NODE_RANK=0
# export MASTER_ADDR=192.168.201.209
# export MASTER_PORT=12346

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT
# export NCCL_IB_CUDA_SUPPORT=1
# export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

# export SPARSITY=24

# export WANDB_ENTITY=eldarkurtic
# export WANDB_DISABLED=False
# export WANDB_PROJECT=llama3_8b_instruct_sp${SPARSITY}

# export MDL="/network/eldar/llama3_8b_instruct/open_platypus/SP2of4_DSETplatypus_train_NSAMPLES1024_PERCDAMP0.05_ISCHATFORMATTrue"

# # export MAX_SEQ_LEN=8192  SET ABOVE FROM TUPLE !
# export PRECISION=amp_bf16
# export USE_FUSED_CROSSENTROPY_LOSS=1

# export MAX_DURATION=1ep
# export EVAL_INTERVAL=1ep

# export GLOBAL_BS=128
# export PER_DEVICE_BS=2

# export LR=1e-4
# export WARMUP=50ba

# # Knowledge distillation
# export TEACHER="meta-llama/Meta-Llama-3-8B-Instruct"
# export KL_TEMP=0.0
# export HARDNESS_KL=0.0
# export HARDNESS_CE=1.0
# export HARDNESS_SQUAREHEAD=1.0

# # GradClipping = YES (maybe check if 1.0 is better than 2.0)
# export RUN_NAME=${DSET}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_GradClip2_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}

# export CLEARML_PROJECT_NAME=eldar_${WANDB_PROJECT}
# export CLEARML_TASK_NAME=${RUN_NAME}

# composer train_sparse_downstream.py \
#     yamls/finetune/llama3_8b_instruct_dset_sweep_GradClip2.yaml \
#     dset_tag=${DSET} \
#     model_name_or_path=${MDL} \
#     max_seq_len=${MAX_SEQ_LEN} \
#     max_duration=${MAX_DURATION} \
#     eval_interval=${EVAL_INTERVAL} \
#     global_train_batch_size=${GLOBAL_BS} \
#     device_train_microbatch_size=${PER_DEVICE_BS} \
#     device_eval_batch_size=${PER_DEVICE_BS} \
#     run_name=${RUN_NAME} \
#     optimizer.lr=${LR} \
#     eval_first=True \
#     scheduler.t_warmup=${WARMUP} \
#     precision=${PRECISION} \
#     knowledge_distillation.teacher_name_or_path=${TEACHER} \
#     knowledge_distillation.temperature=${KL_TEMP} \
#     knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
#     knowledge_distillation.hardness_ce=${HARDNESS_CE} \
#     knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD} \
#     dist_timeout=10000000 2>&1 | tee -a /home/eldar/tmp_logs/${RUN_NAME}.log
# # done

# export DSET="dolphin_flan5m"
# export MAX_SEQ_LEN=8192

# find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}
# # Split the tuple into two parts
# # export DSET=$(echo $tuple | cut -d ' ' -f 1)
# # export MAX_SEQ_LEN=$(echo $tuple | cut -d ' ' -f 2)

# export WORLD_SIZE=32
# export NODE_RANK=0
# export MASTER_ADDR=192.168.201.209
# export MASTER_PORT=12346

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT
# export NCCL_IB_CUDA_SUPPORT=1
# export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

# export SPARSITY=24

# export WANDB_ENTITY=eldarkurtic
# export WANDB_DISABLED=False
# export WANDB_PROJECT=llama3_8b_instruct_sp${SPARSITY}

# export MDL="/network/eldar/llama3_8b_instruct/open_platypus/SP2of4_DSETplatypus_train_NSAMPLES1024_PERCDAMP0.05_ISCHATFORMATTrue"

# # export MAX_SEQ_LEN=8192  SET ABOVE FROM TUPLE !
# export PRECISION=amp_bf16
# export USE_FUSED_CROSSENTROPY_LOSS=1

# export MAX_DURATION=1ep
# export EVAL_INTERVAL=1ep

# export GLOBAL_BS=128
# export PER_DEVICE_BS=2

# export LR=1e-4
# export WARMUP=50ba

# # Knowledge distillation
# export TEACHER="meta-llama/Meta-Llama-3-8B-Instruct"
# export KL_TEMP=0.0
# export HARDNESS_KL=0.0
# export HARDNESS_CE=1.0
# export HARDNESS_SQUAREHEAD=1.0

# # GradClipping = YES (maybe check if 1.0 is better than 2.0)
# export RUN_NAME=${DSET}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_GradClip2_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}

# export CLEARML_PROJECT_NAME=eldar_${WANDB_PROJECT}
# export CLEARML_TASK_NAME=${RUN_NAME}

# composer train_sparse_downstream.py \
#     yamls/finetune/llama3_8b_instruct_dset_sweep_GradClip2.yaml \
#     dset_tag=${DSET} \
#     model_name_or_path=${MDL} \
#     max_seq_len=${MAX_SEQ_LEN} \
#     max_duration=${MAX_DURATION} \
#     eval_interval=${EVAL_INTERVAL} \
#     global_train_batch_size=${GLOBAL_BS} \
#     device_train_microbatch_size=${PER_DEVICE_BS} \
#     device_eval_batch_size=${PER_DEVICE_BS} \
#     run_name=${RUN_NAME} \
#     optimizer.lr=${LR} \
#     eval_first=True \
#     scheduler.t_warmup=${WARMUP} \
#     precision=${PRECISION} \
#     knowledge_distillation.teacher_name_or_path=${TEACHER} \
#     knowledge_distillation.temperature=${KL_TEMP} \
#     knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
#     knowledge_distillation.hardness_ce=${HARDNESS_CE} \
#     knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD} \
#     dist_timeout=10000000 2>&1 | tee -a /home/eldar/tmp_logs/${RUN_NAME}.log
# # done

# export DSET="open_hermes_2_5"
# export MAX_SEQ_LEN=4096

# find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}
# # Split the tuple into two parts
# # export DSET=$(echo $tuple | cut -d ' ' -f 1)
# # export MAX_SEQ_LEN=$(echo $tuple | cut -d ' ' -f 2)

# export WORLD_SIZE=32
# export NODE_RANK=0
# export MASTER_ADDR=192.168.201.209
# export MASTER_PORT=12346

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT
# export NCCL_IB_CUDA_SUPPORT=1
# export TORCH_DISTRIBUTED_DEBUG=OFF #INFO, DETAIL

# export SPARSITY=24

# export WANDB_ENTITY=eldarkurtic
# export WANDB_DISABLED=False
# export WANDB_PROJECT=llama3_8b_instruct_sp${SPARSITY}

# export MDL="/network/eldar/llama3_8b_instruct/open_platypus/SP2of4_DSETplatypus_train_NSAMPLES1024_PERCDAMP0.05_ISCHATFORMATTrue"

# # export MAX_SEQ_LEN=8192  SET ABOVE FROM TUPLE !
# export PRECISION=amp_bf16
# export USE_FUSED_CROSSENTROPY_LOSS=1

# export MAX_DURATION=1ep
# export EVAL_INTERVAL=1ep

# export GLOBAL_BS=128
# export PER_DEVICE_BS=2

# export LR=1e-4
# export WARMUP=50ba

# # Knowledge distillation
# export TEACHER="meta-llama/Meta-Llama-3-8B-Instruct"
# export KL_TEMP=0.0
# export HARDNESS_KL=0.0
# export HARDNESS_CE=1.0
# export HARDNESS_SQUAREHEAD=1.0

# # GradClipping = YES (maybe check if 1.0 is better than 2.0)
# export RUN_NAME=${DSET}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_GradClip2_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}

# export CLEARML_PROJECT_NAME=eldar_${WANDB_PROJECT}
# export CLEARML_TASK_NAME=${RUN_NAME}

# composer train_sparse_downstream.py \
#     yamls/finetune/llama3_8b_instruct_dset_sweep_GradClip2.yaml \
#     dset_tag=${DSET} \
#     model_name_or_path=${MDL} \
#     max_seq_len=${MAX_SEQ_LEN} \
#     max_duration=${MAX_DURATION} \
#     eval_interval=${EVAL_INTERVAL} \
#     global_train_batch_size=${GLOBAL_BS} \
#     device_train_microbatch_size=${PER_DEVICE_BS} \
#     device_eval_batch_size=${PER_DEVICE_BS} \
#     run_name=${RUN_NAME} \
#     optimizer.lr=${LR} \
#     eval_first=True \
#     scheduler.t_warmup=${WARMUP} \
#     precision=${PRECISION} \
#     knowledge_distillation.teacher_name_or_path=${TEACHER} \
#     knowledge_distillation.temperature=${KL_TEMP} \
#     knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
#     knowledge_distillation.hardness_ce=${HARDNESS_CE} \
#     knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD} \
#     dist_timeout=10000000 2>&1 | tee -a /home/eldar/tmp_logs/${RUN_NAME}.log
# # done
