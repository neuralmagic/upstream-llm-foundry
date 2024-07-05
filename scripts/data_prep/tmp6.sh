#!/bin/bash

export ELDAR_DEBUG=1

# python3 convert_custom_dataset.py \
#   --dataset chiayewken/flan-cot --splits train --data_files "**/*" \
#   --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_yesWrap/flan_cot \
#   --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#   --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"


# python3 convert_custom_dataset.py \
#   --dataset /nm/drive0/eldar/datasets/winogrande/merged_for_upstream_v2 \
#   --out_root /nm/drive0/eldar/datasets/winogrande/merged_for_upstream_v2_yesWrap --splits train \
#   --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#   --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

# python3 convert_custom_dataset.py \
#   --dataset /nm/drive0/eldar/datasets/hellaswag/merged_for_upstream_v2 \
#   --out_root /nm/drive0/eldar/datasets/hellaswag/merged_for_upstream_v2_yesWrap --splits train \
#   --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#   --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

# python3 convert_custom_dataset.py \
#   --dataset "/home/eldar/openmathinstruct_1/OpenMathInstruct-1" \
#   --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_yesWrap/openmathinstruct_1 --splits train \
#   --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#   --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

# python3 convert_custom_dataset.py \
#   --dataset /nm/drive0/eldar/datasets/ARC-V1-Feb2018-2/merged_for_upstream \
#   --out_root /nm/drive0/eldar/datasets/ARC-V1-Feb2018-2/merged_for_upstream_yesWrap --splits train \
#   --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#   --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

python3 convert_custom_dataset.py \
  --dataset "Locutusque/UltraTextbooks-2.0" \
  --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_noWrap/ultratextbooks2 --splits train \
  --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
  --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"