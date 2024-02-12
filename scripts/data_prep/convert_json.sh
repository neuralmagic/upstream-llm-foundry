
python3 convert_dataset_json.py \
  --path "/root/github/upstream-llm-foundry/scripts/data_prep/open_llm/pretrain_openllm.jsonl" \
  --out_root "/network/eldar/datasets/llama2_7b/openllm/train" \
  --split train \
  --concat_tokens 2048 \
  --tokenizer "meta-llama/Llama-2-7b-hf"
