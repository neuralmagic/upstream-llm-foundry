#!/bin/bash

export ELDAR_DEBUG=1

# opt125m
    # "garage-bAInd/Open-Platypus": open_platypus,              7_976_960 ~ 8M
    # "Open-Orca/OpenOrca": open_orca,                      1_241_903_104 ~ 1.2B
    # "cognitivecomputations/dolphin": dolphin,             FLAN1m = 273_741_824 ~ 270M, FLAN5m = 810_762_240 ~ 810M
    # "teknium/OpenHermes-2.5": open_hermes_2_5,             343_207_936 ~ 340M
    # "jondurbin/bagel-v0.3": bagel_v03, <-- corrupted format
    # "stingning/ultrachat": ultrachat,                         1_355_352_064 ~ 1.3B
    # "cais/mmlu" auxiliary_train                               29_585_408 ~ 29M

# llama2-7b
# cais/mmlu auxiliary_train = 8322 x 4096 = 34_086_912
# garage-bAInd/Open-Platypus = 2157 x 4096 = 8_835_072

python3 convert_custom_dataset.py \
  --dataset nvidia/OpenMathInstruct-1 \
  --out_root /network/eldar/datasets/llama2_7b/seqlen4k_tokenized/open_math_instruct_1 --splits train validation \
  --concat_tokens 4096 --tokenizer meta-llama/Llama-2-7b-hf --no_wrap --compression zstd
