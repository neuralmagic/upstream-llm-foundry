#!/bin/bash

export ELDAR_DEBUG=1

# ARC-Both synthetic data, three difficulties = {young_children, college_students_ scientists}, two generators = {8b, 70b}
# for GENERATOR in 8b 70b;
# do
#     for DIFFICULTY in young_children college_students scientists;
#     do
#         python3 convert_custom_dataset.py \
#             --dataset "/network/eldar/datasets/data_gen/various_difficulties/arcboth/arc_${DIFFICULTY}_llama3_${GENERATOR}_instruct" --splits train --no_wrap \
#             --out_root "/nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_noWrap/arcboth_datagen_wllama3_${GENERATOR}_instruct_difficulty_${DIFFICULTY}" \
#             --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#             --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"
#     done
# done


# Winogrande synthetic data, three difficulties = {young_children, college_students_ scientists}, two generators = {8b, 70b}
# for GENERATOR in 8b 70b;
# do
#     for DIFFICULTY in young_children college_students scientists;
#     do
#         python3 convert_custom_dataset.py \
#             --dataset "/network/eldar/datasets/data_gen/various_difficulties/winogrande/winogrande_${DIFFICULTY}_llama3_${GENERATOR}_instruct" --splits train --no_wrap \
#             --out_root "/nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_noWrap/winogrande_datagen_wllama3_${GENERATOR}_instruct_difficulty_${DIFFICULTY}" \
#             --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#             --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"
#     done
# done

# Hellaswag synthetic data, three difficulties = {young_children, college_students_ scientists}, two generators = {8b, 70b}
for GENERATOR in 8b 70b;
do
    for DIFFICULTY in young_children college_students scientists;
    do
        python3 convert_custom_dataset.py \
            --dataset "/network/eldar/datasets/data_gen/various_difficulties/hellaswag/hellaswag_${DIFFICULTY}_llama3_${GENERATOR}_instruct" --splits train --no_wrap \
            --out_root "/nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_noWrap/hellaswag_datagen_wllama3_${GENERATOR}_instruct_difficulty_${DIFFICULTY}" \
            --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
            --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"
    done
done

# Hellaswag synthetic data
# python3 convert_custom_dataset.py \
#     --dataset "/network/eldar/datasets/data_gen/hellaswag/hellaswag_llama3_8b_instruct" --splits train --no_wrap \
#     --out_root "/nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_noWrap/hellaswag_datagen_wllama3_8b_instruct" \
#     --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#     --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

# python3 convert_custom_dataset.py \
#     --dataset "/network/eldar/datasets/data_gen/hellaswag/hellaswag_llama3_70b_instruct" --splits train --no_wrap \
#     --out_root "/nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_noWrap/hellaswag_datagen_wllama3_70b_instruct" \
#     --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#     --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

# MMLU synthetic data
# python3 convert_custom_dataset.py \
#     --dataset "/network/eldar/datasets/data_gen/mmlu" --data_files "mmlu_llama3_8b_instruct/**/*.jsonl" --splits train --no_wrap \
#     --out_root "/nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_noWrap/mmlu_datagen_wllama3_8b_instruct" \
#     --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#     --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

# python3 convert_custom_dataset.py \
#     --dataset "/network/eldar/datasets/data_gen/mmlu" --data_files "mmlu_llama3_70b_instruct/**/*.jsonl" --splits train --no_wrap \
#     --out_root "/nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_noWrap/mmlu_datagen_wllama3_70b_instruct" \
#     --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#     --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"


# OpenMathInstruct1 synthetic data
# python3 convert_custom_dataset.py \
#     --dataset "/network/eldar/datasets/data_gen/openmathinstruct1" --data_files "openmathinstruct1_llama3_8b_instruct/**/*.jsonl" --splits train --no_wrap \
#     --out_root "/nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_noWrap/openmathinstruct1_datagen_wllama3_8b_instruct" \
#     --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#     --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

# python3 convert_custom_dataset.py \
#     --dataset "/network/eldar/datasets/data_gen/openmathinstruct1" --data_files "openmathinstruct1_llama3_70b_instruct/**/*.jsonl" --splits train --no_wrap \
#     --out_root "/nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_noWrap/openmathinstruct1_datagen_wllama3_70b_instruct" \
#     --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#     --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

# python3 convert_custom_dataset.py \
#     --dataset "/network/eldar/datasets/shubhra_deduplicated" --data_files "downstream_mix/train_split_part*.jsonl" --splits train --no_wrap \
#     --out_root "/nm/drive0/eldar/datasets/llama3_8b/seqlen8k_tokenized_wEOS_noWrap/downstream_mix_shubhra_deduplicated" \
#     --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#     --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"
