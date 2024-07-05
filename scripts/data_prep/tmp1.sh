#!/bin/bash

export ELDAR_DEBUG=1

# --no_wrap
python3 convert_custom_dataset.py \
    --dataset garage-bAInd/Open-Platypus --splits train \
    --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_yesWrap/open_platypus \
    --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
    --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

python3 convert_custom_dataset.py \
    --dataset Open-Orca/OpenOrca --splits train \
    --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_yesWrap/open_orca \
    --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
    --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

python3 convert_custom_dataset.py \
    --dataset teknium/OpenHermes-2.5 --splits train \
    --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_yesWrap/open_hermes_2_5 \
    --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
    --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

python3 convert_custom_dataset.py \
    --dataset stingning/ultrachat --splits train \
    --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_yesWrap/ultrachat \
    --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
    --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

python3 convert_custom_dataset.py \
    --dataset yahma/alpaca-cleaned --splits train \
    --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_yesWrap/alpaca_cleaned \
    --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
    --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

python3 convert_custom_dataset.py \
    --dataset mosaicml/dolly_hhrlhf --splits train \
    --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_yesWrap/dolly_hhrlhf \
    --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
    --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

python3 convert_custom_dataset.py \
    --dataset /home/eldar/datasets/fineweb_sample10BT --splits train \
    --out_root /home/eldar/datasets/fineweb_sample10BT/llama3_8b_FineWebSample10BT_seq8k_tokenized_wEOS_yesWrap \
    --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
    --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"
