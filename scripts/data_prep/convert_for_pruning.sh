#!/bin/bash

export ELDAR_DEBUG=1

python convert_dataset_hf.py \
  --dataset c4 --data_subset en \
  --out_root /home/eldar/llama3_c4_seqlen2048_wEOS --splits val_xsmall \
  --concat_tokens 2048 --tokenizer meta-llama/Meta-Llama-3-8B --no_wrap \
  --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"
