

for X in denseLlama2_7b_2ep_lr3e-5_bs32_GradClip1_warmup20ba denseLlama2_7b_2ep_lr5e-5_bs32_GradClip1_warmup20ba denseLlama2_7b_4ep_lr3e-5_bs32_GradClip1_warmup20ba denseLlama2_7b_4ep_lr5e-5_bs32_GradClip1_warmup20ba;
do
    export COMPOSER_PATH="/home/eldar/llmfoundry_checkpoints/llama2_7b_mathador/${X}/latest-rank0.pt"
    export HF_OUTPUT_PATH="/home/eldar/llmfoundry_checkpoints/llama2_7b_mathador/${X}/hf"
    python convert_composer_to_hf.py --composer_path $COMPOSER_PATH --hf_output_path $HF_OUTPUT_PATH --output_precision bf16
done

