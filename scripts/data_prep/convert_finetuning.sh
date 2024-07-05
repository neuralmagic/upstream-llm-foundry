#!/bin/bash

    # --data_subset main \

export SEQLEN=1024

python convert_finetuning_dataset.py \
    --dataset "cognitivecomputations/dolphin" \
    --data_subset "flan1m-alpaca-uncensored" \
    --data_files "flan1m-alpaca-uncensored-deduped.jsonl" \
    --splits "train" \
    --max_seq_len ${SEQLEN} \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --out_root "/home/eldar/datasets/llama2_7b/dolphin_flan1m_deduped_seqlen${SEQLEN}"
