#!/bin/bash

# /network/alexandre/research/cerebras/llama2_7B_sparse50_retrained
# /network/alexandre/research/cerebras/llama2_450M_base

export DATA_LOCAL=/network/eldar/datasets/opt125m/c4

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=test_evals

export USE_FUSED_CROSSENTROPY_LOSS=1

export MAX_SEQ_LEN=4096

export PER_DEVICE_BS=16
export RUN_NAME=test

composer eval.py \
    yamls/eval.yaml \
    max_seq_len=${MAX_SEQ_LEN} \
    device_eval_batch_size=${PER_DEVICE_BS} \
    run_name=${RUN_NAME} 2>&1 | tee -a icl_evals_ppl_ablation_part4.txt

