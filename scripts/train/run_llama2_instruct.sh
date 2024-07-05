#!/bin/bash

# model = sp70 only
# Nmgpu17 = 3ep x 1e-5
# Nmgpu18 = 3ep x 3e-5
# Nmgpu19 = 3ep x 5e-5
# Nmgpu20 = 2ep x {1e-5, 3e-5}
# Nmgpu11 = 1ep x {1e-5, 3e-5, 5e-5}
# Nmgpu12 = 2ep x 5e-5


for LR in 5e-5;
do
    for SPARSITY in 70;
    do
        for MAX_DURATION in 2ep;
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

        export GLOBAL_BS=144
        export PER_DEVICE_BS=18

        export WARMUP=200ba

            export TEACHER=/network/eldar/llama2_7b_platydolphin1m/nm/NMmdl_sp0_uniform_noKD_2ep_lr1e-5_bs512_noGradClip_warmup100ba_seqlen1024
            export KL_TEMP=0.0
            export HARDNESS_KL=0.0
            export HARDNESS_CE=1.0
            export HARDNESS_SQUAREHEAD=1.0

            export RUN_NAME=NMmdl_CE${HARDNESS_CE}_SquareHead${HARDNESS_SQUAREHEAD}_sp${SPARSITY}_uniform_${MAX_DURATION}_lr${LR}_bs${GLOBAL_BS}_noGradClip_warmup${WARMUP}_seqlen${MAX_SEQ_LEN}

            composer train_sparse_downstream.py \
                yamls/finetune/platy+dolphin1m_noGradClip_yesKD_cosineLR.yaml \
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
                knowledge_distillation.hardness_squarehead=${HARDNESS_SQUAREHEAD}
        done
    done
done
