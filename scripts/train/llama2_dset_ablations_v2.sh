#!/bin/bash

# TODO: dset stats
# openllm_test_stories = 2571 x 4096 = 10_530_816 = 10.5M                    <-- 1ep = 0.1h    <-- 10ep
# gsm8k = 348 x 4096 = 1_425_408 = 1.4M                                      <-- 1ep = 0.01h   <-- 10ep
# garage-bAInd/Open-Platypus = 2157 x 4096 = 8_835_072 = 8.8M                <-- 1ep = 0.1h    <-- 10ep
# cais/mmlu auxiliary_train = 8322 x 4096 = 34_086_912 = 34M                 <-- 1ep = 0.3h    <-- 10ep
# X nvidia/OpenMathInstruct-1 validation = 13230 x 4096 = 54_190_080 = 54M     <-- 1ep = 0.5h    <-- 5ep
# teknium/OpenHermes-2.5 = 95213 x 4096 = 389_992_448 = 340M                 <-- 1ep = 3.4h
# nvidia/OpenMathInstruct-1 train = 86500 x 4096 = 354_304_000 = 354M        <-- 1ep = 3.5h
# cognitivecomputations/dolphin, flan1m = 79935 x 4096 = 327_413_760 = 330M  <-- 1ep = 3.3h

# cognitivecomputations/dolphin, flan5m = 238952 x 4096 = 978_747_392 = 980M <-- 1ep = 10h
# stingning/ultrachat = 429449 x 4096 = 1_759_023_104 = 1.7B                 <-- 1ep = 17h
# Open-Orca/OpenOrca = 364815 x 4096 = 1_494_282_240 = 1.5B                  <-- 1ep = 15h

# TODO: for multi-node runs
# export WORLD_SIZE=24
# export NODE_RANK=0
# export MASTER_ADDR=192.168.201.209
# export MASTER_PORT=12345
# export NCCL_DEBUG=INFO

datasets=(
    # "openllm_test_set_stories:10"
    # "mmlu_auxiliary_train:10"
    # "open_platypus:10"
    # "gsm8k:10"
    # "open_math_instruct_1:1"
    # "open_hermes_2_5:1"
    # "dolphin_flan1m:1"
    "open_orca:2"
    "dolphin_flan5m:2"
    #"ultrachat:1"
)

# Iterate over datasets
for dataset in "${datasets[@]}"; do
    # Split dataset name and epochs using ':' as delimiter
    IFS=':' read -r name epochs_str <<< "$dataset"

    # Split epochs into an array using ' ' (space) as delimiter
    IFS=' ' read -ra epochs <<< "$epochs_str"

    # Iterate over epochs
    for ep in "${epochs[@]}"; do
        echo "Processing dataset $name with epoch $ep"
# ==============================================================================
        export DATA_LOCAL=/network/eldar/datasets/llama2_7b/seqlen4k_tokenized/${name}
        export SPARSITY=70
        export MDL_TAG=oneshot_sparsegpt_sp${SPARSITY}_nsamples512_seqlen2048  # TODO: for checkpoints only
        export DSET_TAG=${name}  # TODO: for checkpoints only

        export WANDB_ENTITY=eldarkurtic
        export WANDB_DISABLED=False
        export WANDB_PROJECT=llama2_7b_sp${SPARSITY}_${DSET_TAG}

        export MDL=/network/eldar/llama2_7b_c4/${MDL_TAG}
        export MAX_SEQ_LEN=4096
        export PRECISION=amp_bf16
        export USE_FUSED_CROSSENTROPY_LOSS=1

        export MAX_DURATION=${ep}ep
        export EVAL_INTERVAL=500ba

        export GLOBAL_BS=128
        export PER_DEVICE_BS=16

        export LR=3e-4
        export WARMUP=100ba

        # TODO: no KD for dset ablations
        # Knowledge distillation
        # export TEACHER=meta-llama/Llama-2-7b-hf
        # export KL_TEMP=0.0
        # export HARDNESS_KL=0.0
        # export HARDNESS_CE=1.0
        # export HARDNESS_SQUAREHEAD=1.0

        export RUN_NAME=${MDL_TAG}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_noGradClip_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}_fusedCE${USE_FUSED_CROSSENTROPY_LOSS}

        composer train_sparse.py \
            yamls/pretrain/llama2_7b_dset_ablations.yaml \
            model_name_or_path=${MDL} \
            max_seq_len=${MAX_SEQ_LEN} \
            data_local=${DATA_LOCAL} \
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
            model_tag=llama2_7b_${MDL_TAG} \
            dset_tag=${DSET_TAG}
            # knowledge_distillation.teacher_name_or_path=${TEACHER} \
            # knowledge_distillation.temperature=${KL_TEMP} \
            # knowledge_distillation.hardness_kldiv=${HARDNESS_KL} \
            # knowledge_distillation.hardness_ce=${HARDNESS_CE} \
        # knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD}
# ==============================================================================
    done
done
