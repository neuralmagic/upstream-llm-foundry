#!/bin/bash

#git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B /home/eldar/llama2_7b_SlimPajama_seq4k/ &&
export ELDAR_DEBUG=1

python3 convert_custom_dataset.py \
  --dataset /home/eldar/SlimPajama-627B \
  --out_root /home/eldar/datasets/llama2_7b_SlimPajama_seq4k_tokenized --splits train validation \
  --concat_tokens 4096 --tokenizer meta-llama/Llama-2-7b-hf --no_wrap --compression zstd

