#!/bin/bash

#git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B /home/eldar/llama2_7b_SlimPajama_seq4k/ &&
export ELDAR_DEBUG=1

# old Llama-3 where we add BOS token to split concatenated sequences
# python3 convert_custom_dataset.py \
#   --dataset /nm/drive0/eldar/datasets/cosmopedia --data_files "data/**/*" --splits train \
#   --out_root /nm/drive0/eldar/datasets/llama3_8b_Cosmopedia_seq8k_tokenized \
#   --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --no_wrap --compression zstd

# new Llama-3 where we add EOS token to split concatenated sequences
# NOTE: with tokenizer_call_kwargs we disable BOS/EOS completely, and with eos_text we add it manually
# python3 convert_custom_dataset.py \
#   --dataset /nm/drive0/eldar/datasets/cosmopedia --data_files "data/**/*" --splits train \
#   --out_root /nm/drive0/eldar/datasets/llama3_8b_Cosmopedia_seq8k_tokenized_wEOS_yesWrap \
#   --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
#   --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"

# NOTE: here we tokenize each subset independently
# Cosmopedia subsets = {auto_math_text, khanacademy, openstax, stanford, stories, web_samples_v1, web_samples_v2, wikihow}
for SUBSET in auto_math_text khanacademy openstax stanford stories web_samples_v1 web_samples_v2 wikihow;
do
    python3 convert_custom_dataset.py \
    --dataset /nm/drive0/eldar/datasets/cosmopedia --data_subset ${SUBSET} --splits train \
    --out_root llama3_8b_Cosmopedia_seq8k_tokenized_wEOS_noWrap_subsets/${SUBSET} --no_wrap \
    --concat_tokens 8192 --tokenizer meta-llama/Meta-Llama-3-8B --compression zstd \
    --eos_text "<|end_of_text|>" --tokenizer_call_kwargs "{\"add_special_tokens\": false}"
done