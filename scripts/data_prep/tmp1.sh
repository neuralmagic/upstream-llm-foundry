#!/bin/bash

export ELDAR_DEBUG=1

    # "garage-bAInd/Open-Platypus": open_platypus,              7_976_960 ~ 8M
    # "Open-Orca/OpenOrca": open_orca,                      1_241_903_104 ~ 1.2B
    # "cognitivecomputations/dolphin": dolphin,             FLAN1m = 273_741_824 ~ 270M, FLAN5m = 810_762_240 ~ 810M
    # "teknium/OpenHermes-2.5": open_hermes_2_5,             343_207_936 ~ 340M
    # "jondurbin/bagel-v0.3": bagel_v03, <-- corrupted format
    # "stingning/ultrachat": ultrachat,                         1_355_352_064 ~ 1.3B
    # "cais/mmlu" auxiliary_train                               29_585_408 ~ 29M

python3 convert_custom_dataset.py \
  --dataset cais/mmlu \
  --data_subset auxiliary_train \
  --out_root /root/datasets/opt125m_seq2k/mmlu --splits train \
  --concat_tokens 2048 --tokenizer facebook/opt-125m --no_wrap --compression zstd

