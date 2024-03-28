#!/bin/bash

export ELDAR_DEBUG=1

python3 convert_custom_dataset.py \
  --dataset Open-Orca/OpenOrca --splits train --eos_test "<|end_of_text|>" \
  --out_root /nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized/open_orca \
  --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --no_wrap --compression zstd
    # "garage-bAInd/Open-Platypus": open_platypus,
    # "Open-Orca/OpenOrca": open_orca,
    # "cognitivecomputations/dolphin": dolphin,
    # "teknium/OpenHermes-2.5": open_hermes_2_5,
    # "jondurbin/bagel-v0.3": bagel_v03,
    # "stingning/ultrachat": ultrachat,

# python3 convert_custom_dataset.py \
#   --dataset cognitivecomputations/dolphin \
#   --out_root /root/datasets/opt125m_seq2k/dolphin --splits train \
#   --data_subset flan5m-alpaca-uncensored \
#   --concat_tokens 2048 --tokenizer facebook/opt-125m --no_wrap --compression zstd


python3 convert_custom_dataset.py \
  --dataset /network/eldar/datasets/ARC-V1-Feb2018-2/for_llmfoundry \
  --out_root /root/arc_corpus --splits train \
  --concat_tokens 4096 --tokenizer meta-llama/Llama-2-7b-hf --no_wrap --compression zstd
