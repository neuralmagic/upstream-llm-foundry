#!/bin/bash

    # "garage-bAInd/Open-Platypus": open_platypus,              7_976_960 ~ 8M    => 3_895 / 256 = 15.2 steps
    # "Open-Orca/OpenOrca": open_orca,                      1_241_903_104 ~ 1.2B  => 606_398 / 256 = 2_370 steps
    # "cognitivecomputations/dolphin": dolphin,             FLAN1m = 273_741_824 ~ 270M => 133_663 / 256 = 522, FLAN5m = 810_762_240 ~ 810M => 395_880 / 256 = 1546
    # "teknium/OpenHermes-2.5": open_hermes_2_5,             343_207_936 ~ 340M => 167_582 = 654
    # "stingning/ultrachat": ultrachat,                         1_355_352_064 ~ 1.3B => 661_793 / 256 = 2585
    # open_llm_synthetic  9_009_152 ~ 9M => 4_399 / 256 = 17.2

datasets=(
    "open_llm_synthetic:100"
    # "open_platypus:1 3 5"
    # "open_orca:1 3"
    # "dolphin-flan1M:1 3 5"
    # "dolphin-flan5M:1 3"
    # "open_hermes_2_5:1 3 5"
    # "ultrachat:1 3"
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
        export DATA_LOCAL=/root/datasets/opt125m_seq2k/${name}

        export WANDB_ENTITY=eldarkurtic
        export WANDB_DISABLED=False
        export WANDB_PROJECT=opt125m_${name}

        export USE_FUSED_CROSSENTROPY_LOSS=1
        export MDL=facebook/opt-125m
        export MAX_SEQ_LEN=2048
        export PRECISION=amp_bf16

        export MAX_DURATION=${ep}ep
        export EVAL_INTERVAL=1ep

        export GLOBAL_BS=256
        export PER_DEVICE_BS=32

        export LR=1e-4
        export WARMUP=10ba

        export RUN_NAME=dense_${name}_${PRECISION}_maxseq${MAX_SEQ_LEN}_${MAX_DURATION}_cosineLR${LR}_warmup${WARMUP}_yesGradClip1.0_globalBS${GLOBAL_BS}_evalInterval${EVAL_INTERVAL}_fusedCE${USE_FUSED_CROSSENTROPY_LOSS}

        composer train.py \
            yamls/pretrain/opt125m_openllm.yaml \
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
            precision=${PRECISION}
# ==============================================================================
    done
done
