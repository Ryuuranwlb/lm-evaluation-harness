MODEL_PATH=${1:-/home/u20140041/code/HALOs/data/models/yulan_dummy-dpo-para-1/FINAL}

CUDA_VISIBLE_DEVICES=6 lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks gsm8k_cot_qwen \
    --gen_kwargs max_gen_toks=2048 \
    --batch_size 256 \
    --apply_chat_template \
    --output_path model_result \
    --log_samples


CUDA_VISIBLE_DEVICES=6 lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True,gpu_memory_utilization=0.9 \
    --tasks hendrycks_math_yulan \
    --apply_chat_template \
    --batch_size 512 \
    --gen_kwargs max_gen_toks=2048 \
    --output_path model_result \
    --log_samples 