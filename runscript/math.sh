MODEL_PATH=${1:-/media/yulan/sft/mix_3M_no_multi-12-21}

CUDA_VISIBLE_DEVICES=2 lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks gsm8k_cot_qwen \
    --gen_kwargs max_gen_toks=2048 \
    --batch_size 256 \
    --apply_chat_template 


CUDA_VISIBLE_DEVICES=2 lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True,gpu_memory_utilization=0.9 \
    --tasks hendrycks_math_yulan \
    --apply_chat_template \
    --batch_size 512 \
    --gen_kwargs max_gen_toks=2048 \
    --output_path result \
    --log_samples 