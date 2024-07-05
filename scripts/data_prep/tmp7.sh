#!/bin/bash

export ELDAR_DEBUG=1

python3 convert_custom_dataset.py \
    --dataset HuggingFaceFW/fineweb-edu --splits train --no_wrap --data_subset sample-10BT \
    --out_root /home/eldar/datasets/fineweb_edu_sample10BT/llama3_8b_FineWebEduSample10BT_seq8k_tokenized_wEOS_noWrap \
    --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
    --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"
