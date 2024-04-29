#!/bin/bash

export WANDB_ENTITY=eldarkurtic
export WANDB_DISABLED=False
export WANDB_PROJECT=test_evals

export USE_FUSED_CROSSENTROPY_LOSS=1
export MAX_SEQ_LEN=2048
export PER_DEVICE_BS=16

export BASE_PATH=/network/eldar
export PROJECT=cerebras
export RUN_NAME=openllm_seqlen${MAX_SEQ_LEN}_perdeviceBS${PER_DEVICE_BS}_worldsize4

#llama2-7b_dolphin+open-platypus_transfer-pruned70;
for CKPT in Llama-2-7b-pruned50-retrained;
do
    echo $CKPT
    mkdir -p eval/${PROJECT}/${CKPT}

    composer eval/eval.py \
        eval/yamls/eval_openllm.yaml \
        max_seq_len=${MAX_SEQ_LEN} \
        device_eval_batch_size=${PER_DEVICE_BS} \
        ckpt_path=${BASE_PATH}/${PROJECT}/${CKPT} 2>&1 | tee eval/${PROJECT}/${CKPT}/${RUN_NAME}.txt
done
