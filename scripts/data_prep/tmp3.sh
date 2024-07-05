#!/bin/bash

export ELDAR_DEBUG=1

python3 convert_custom_dataset.py \
    --dataset prigoyal/flan2021_submix_original --splits train \
    --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_yesWrap/flan2021_submix_original \
    --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
    --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"