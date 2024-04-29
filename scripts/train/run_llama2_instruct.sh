#!/bin/bash

# 3xA100 + 4xH100 = 7 servers
# 8-gpus, no KD, batch-size=112

# LR = {1e-4, 8e-5, 5e-5}
# models = {sp24, sp70}
# <-- put sp24+LR=1e-4 on 1xH100s, sp24+otherLRs on 2xA100s, sp70 on 3xH100s -->
# num_epochs = {1, 2, 3, 4} ~ 60 hours

export LR=5e-5

for SPARSITY in 24;
do
    for MAX_DURATION in 1ep 2ep 3ep 4ep;
    do
        find /tmp -type d -name "train" 2>/dev/null | xargs -I{} rm -rf {}

        export WANDB_ENTITY=eldarkurtic
        export WANDB_DISABLED=False
        export WANDB_PROJECT=llama2_platy+dolphin1m_sp${SPARSITY}

        export USE_FUSED_CROSSENTROPY_LOSS=1
        export MDL=/network/eldar/models_to_share/llama2_7b_sp${SPARSITY}_v1

        export MAX_SEQ_LEN=1024
        export PRECISION=amp_bf16
        export EVAL_INTERVAL=1ep

        export GLOBAL_BS=512
        export PER_DEVICE_BS=64

        export WARMUP=50ba

        #export RUN_NAME=NMmdl_CE${HARDNESS_CE}_SquareHead${HARDNESS_SQUAREHEAD}_oneshot_sp${SPARSITY}_uniform_${MAX_DURATION}_lr${LR}_bs${GLOBAL_BS}_noGradClip_warmup${WARMUP}
        export RUN_NAME=NMmdl_sp${SPARSITY}_uniform_noKD_${MAX_DURATION}_lr${LR}_bs${GLOBAL_BS}_noGradClip_warmup${WARMUP}_seqlen${MAX_SEQ_LEN}

        composer train_sparse_downstream.py \
            yamls/finetune/platy+dolphin1m_noGradClip_noKD_cosineLR.yaml \
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
            precision=${PRECISION}
    done
done