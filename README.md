# lm-evaluation-harness

## Setup

```bash
# Create and activate the venv
uv venv .venv --python 3.11
source .venv/bin/activate

# Install the package with vllm backend
uv pip install -e ".[vllm,hf,math,ifeval]"
```

## Evaluation

`mgsm_rev2_native_cot_fr`, `gpqa_diamond_fr_cot`, `global_mmlu_fr_cot`, `math500_multilingual_french`, `aime24_multilingual_fr`, `humanevalplus_multilingual_fr` and `aime25_multilingual_fr`

```bash
CUDA_VISIBLE_DEVICES=0,1 \
HF_ALLOW_CODE_EVAL=1 \
nohup lm_eval \
    --model vllm \
    --model_args "pretrained=LiquidAI/LFM2.5-1.2B-Instruct,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.7,max_model_len=16384,enable_thinking=False" \
    --apply_chat_template \
    --tasks mgsm_rev2_native_cot_fr,gpqa_diamond_fr_cot,global_mmlu_fr_cot,math500_multilingual_french,aime24_multilingual_fr,aime25_multilingual_fr,humanevalplus_multilingual_fr \
    --batch_size auto \
    --gen_kwargs do_sample=False,temperature=0,max_gen_toks=15000 \
    --confirm_run_unsafe_code \
    --output_path eval_results/french_eval_result \
    --log_samples \
    --n_runs 10 \
    > logs/french_bench.log 2>&1 &
```
