
for X in DownFixedFull_SYN8BwdiffsOpenLLM4x_SYN8Bopenmath1x_FineWebEdu10BT_oneshot_wEOS_sp24_amp_bf16_maxseq8192_1ep_cosineLR1e-4_warmup600ba_GradClip2_globalBS128_evalInterval6000ba_wEOS_noWrap;
do
    export COMPOSER_PATH="/home/eldar/llmfoundry_checkpoints/llama3_8b_cosmopedia_alldownstream/${X}/latest-rank0.pt"
    export HF_OUTPUT_PATH="/home/eldar/llmfoundry_checkpoints/llama3_8b_cosmopedia_alldownstream/${X}/hf"
    python convert_composer_to_hf.py --composer_path $COMPOSER_PATH --hf_output_path $HF_OUTPUT_PATH --output_precision bf16
done

