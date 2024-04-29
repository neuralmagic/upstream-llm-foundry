
for X in NMmdl_CE1.0_SquareHead1.0_oneshot_sp24_uniform_2ep_lr1e-4_bs32_noGradClip_warmup20ba NMmdl_CE1.0_SquareHead1.0_oneshot_sp24_uniform_3ep_lr1e-4_bs32_noGradClip_warmup20ba NMmdl_CE1.0_SquareHead1.0_oneshot_sp24_uniform_4ep_lr1e-4_bs32_noGradClip_warmup20ba NMmdl_CE1.0_SquareHead1.0_oneshot_sp70_uniform_2ep_lr1e-4_bs32_noGradClip_warmup20ba NMmdl_CE1.0_SquareHead1.0_oneshot_sp70_uniform_3ep_lr1e-4_bs32_noGradClip_warmup20ba NMmdl_CE1.0_SquareHead1.0_oneshot_sp70_uniform_4ep_lr1e-4_bs32_noGradClip_warmup20ba;
do
    export COMPOSER_PATH="/nm/drive0/eldar/llmfoundry_checkpoints/llama2_7b_gsm8k/${X}/latest-rank0.pt"
    export HF_OUTPUT_PATH="/nm/drive0/eldar/llmfoundry_checkpoints/llama2_7b_gsm8k/${X}/hf"
    python convert_composer_to_hf.py --composer_path $COMPOSER_PATH --hf_output_path $HF_OUTPUT_PATH --output_precision bf16
done
