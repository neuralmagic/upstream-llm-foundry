#!/bin/bash

# export HF_DATASETS_CACHE="/localhome/ekurtic/hf_cache"

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=test_evals

export USE_FUSED_CROSSENTROPY_LOSS=1
export MAX_SEQ_LEN=2048
export PER_DEVICE_BS=16

export BASE_PATH=/network/eldar
export PROJECT=cerebras
export RUN_NAME=openllm_seqlen${MAX_SEQ_LEN}_perdeviceBS${PER_DEVICE_BS}_worldsize4

export BASE_PATH=/nfs/scistore19/alistgrp/ekurtic/eldar-upstream/llmfoundry_checkpoints
export PROJECT=llama2_7b_cosmopedia_alldownstream
export RUN_NAME=gauntletv03_seqlen${MAX_SEQ_LEN}_perdeviceBS${PER_DEVICE_BS}_worldsize4

for CKPT in oneshot_sp70_fromConverged24_fromConverged70onDolma_amp_bf16_maxseq4096_1ep_cosineLR1e-4_warmup1500ba_noGradClip_globalBS256_evalInterval10000ba;
do
    echo $CKPT
    mkdir -p eval/${PROJECT}/${CKPT}

    composer eval/eval.py \
        eval/yamls/eval_openllm.yaml \
        max_seq_len=${MAX_SEQ_LEN} \
        device_eval_batch_size=${PER_DEVICE_BS} \
        ckpt_path=${BASE_PATH}/${PROJECT}/${CKPT} 2>&1 | tee eval/${PROJECT}/${CKPT}/${RUN_NAME}.txt
done
