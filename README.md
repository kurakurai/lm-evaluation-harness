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

`mgsm_rev2_native_cot_fr`, `gpqa_diamond_fr_cot`, `global_mmlu_fr_cot`, `math500_multilingual_french`, `aime24_multilingual_fr`, `humanevalplus_multilingual_fr`, `aime25_multilingual_fr`, `ifeval_fr`, `global_piqa_prompted_fra_latn_fran`, `belebele_fra_Latn_generative` and `mmlu_prox_lite_fr_generative`

```bash
CUDA_VISIBLE_DEVICES=6,7 \
HF_ALLOW_CODE_EVAL=1 \
nohup lm_eval \
    --model vllm \
    --model_args "pretrained=Qwen/Qwen3.5-0.8B,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.7,max_model_len=16384,enable_thinking=False" \
    --apply_chat_template \
    --tasks mgsm_rev2_native_cot_fr,gpqa_diamond_fr_cot,global_mmlu_fr_cot,math500_multilingual_french,aime24_multilingual_fr,aime25_multilingual_fr,humanevalplus_multilingual_fr,ifeval_fr,global_piqa_prompted_fra_latn_fran,belebele_fra_Latn_generative,mmlu_prox_lite_fr_generative \
    --batch_size auto \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,top_k=20,max_gen_toks=15000 \
    --confirm_run_unsafe_code \
    --output_path eval_results/french_eval_result \
    --log_samples \
    --n_runs 10 \
    > logs/qwen3.5-0.8B_bench.log 2>&1 &
```
