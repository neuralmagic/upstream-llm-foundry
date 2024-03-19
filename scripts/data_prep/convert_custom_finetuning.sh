#!/bin/bash

export ELDAR_DEBUG=1

    # "garage-bAInd/Open-Platypus": open_platypus,
    # "Open-Orca/OpenOrca": open_orca,
    # "cognitivecomputations/dolphin": dolphin,
    # "teknium/OpenHermes-2.5": open_hermes_2_5,
    # "jondurbin/bagel-v0.3": bagel_v03,
    # "stingning/ultrachat": ultrachat,

python3 convert_custom_dataset.py \
  --dataset cognitivecomputations/dolphin \
  --out_root /root/datasets/opt125m_seq2k/dolphin --splits train \
  --data_subset flan5m-alpaca-uncensored \
  --concat_tokens 2048 --tokenizer facebook/opt-125m --no_wrap --compression zstd

