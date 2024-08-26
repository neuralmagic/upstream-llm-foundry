#!/bin/bash

export ELDAR_DEBUG=0

python convert_dataset_hf.py \
  --dataset c4 --data_subset en \
  --out_root /home/eldar/c4_dset_for_llama3_8k_seq_len_wrap --splits train_small val_small \
  --concat_tokens 8096 --tokenizer meta-llama/Meta-Llama-3-8B-Instruct
