#!/bin/bash

export ELDAR_DEBUG=1

python3 convert_custom_dataset.py \
  --dataset Open-Orca/OpenOrca --splits train --eos_test "<|end_of_text|>" \
  --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized/open_orca \
  --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --no_wrap --compression zstd