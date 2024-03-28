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
# cais/mmlu auxiliary_train = 8322 x 4096 = 34_086_912 = 34M
# garage-bAInd/Open-Platypus = 2157 x 4096 = 8_835_072 = 8.8M
# teknium/OpenHermes-2.5 = 95213 x 4096 = 389_992_448 = 340M
# stingning/ultrachat = 429449 x 4096 = 1_759_023_104 = 1.7B
# Open-Orca/OpenOrca = 364815 x 4096 = 1_494_282_240 = 1.5B
# cognitivecomputations/dolphin, flan5m = 238952 x 4096 = 978_747_392 = 980M
# cognitivecomputations/dolphin, flan1m = 79935 x 4096 = 327_413_760 = 330M
# nvidia/OpenMathInstruct-1 train = 86500 x 4096 = 354_304_000 = 354M
# nvidia/OpenMathInstruct-1 validation = 13230 x 4096 = 54_190_080 = 54M
# gsm8k = 348 x 4096 = 1_425_408 = 1.4M
# openllm_test_stories = 2571 x 4096 = 10_530_816 = 10.5M

python3 convert_custom_dataset.py \
  --dataset prigoyal/flan2021_submix_original \
  --out_root /network/eldar/datasets/llama2_7b/seqlen4k_tokenized/flan2021_submix_original --splits train \
  --concat_tokens 4096 --tokenizer meta-llama/Llama-2-7b-hf --no_wrap --compression zstd