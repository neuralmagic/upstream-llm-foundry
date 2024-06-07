#!/bin/bash

# export HF_DATASETS_CACHE="/localhome/ekurtic/hf_cache"

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=test_evals

export USE_FUSED_CROSSENTROPY_LOSS=1
export MAX_SEQ_LEN=4096
export PER_DEVICE_BS=8

export PROJECT=sparse_llama3_8b
export RUN_NAME=evalgauntletv03nogsm8k_seqlen${MAX_SEQ_LEN}_perdeviceBS${PER_DEVICE_BS}_worldsize4

export CKPT=DownFixedMathInstCosmoFineWebBoth10BT_oneshot_wEOS_sp24_amp_bf16_maxseq8192_1ep_cosineLR1e-4_warmup1000ba_GradClip2_globalBS128_evalInterval10000ba_wEOS_noWrap
echo $CKPT
mkdir -p eval/results/${PROJECT}/${CKPT}

composer eval/eval.py \
    eval/yamls/eval_evalgauntletv03_noGSM8k.yaml \
    max_seq_len=${MAX_SEQ_LEN} \
    device_eval_batch_size=${PER_DEVICE_BS} \
    ckpt_path=/network/eldar/llmfoundry_checkpoints/llama3_8b_cosmopedia_alldownstream/${CKPT} 2>&1 | tee eval/results/${PROJECT}/${CKPT}/${RUN_NAME}.txt
