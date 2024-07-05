#!/bin/bash

export ELDAR_DEBUG=1

python3 convert_custom_dataset.py \
    --dataset cognitivecomputations/dolphin --splits train --data_subset flan1m-alpaca-uncensored \
    --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_yesWrap/dolphin_flan1m \
    --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
    --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"
