# Multi-IF (French)

Faithful French subset of [Multi-IF](https://huggingface.co/datasets/facebook/Multi-IF)
([paper](https://arxiv.org/abs/2410.15553), [official code](https://github.com/facebookresearch/Multi-IF)) —
a **multi-turn** extension of IFEval. Each conversation has 3 turns; the model's own
response to turn *N* is fed back as context for turn *N+1*, and each turn is scored
against the **cumulative** list of instructions introduced so far.

## Why this is a standalone runner, not a `--tasks` entry

The harness collects every request for a task and sends them to the model in a single
batched call (`lm_eval/evaluator.py`), so a task cannot build turn *N+1* from a response
it has not generated yet. Faithful multi-turn therefore needs its own generation loop.
This runner reuses the harness's model registry, chat templating, generation, and the
IFEval verifiers, so the launch experience matches the other French tasks.

The French split has **548 conversations**. Every instruction id it uses is already
implemented in `lm_eval/tasks/ifeval/instructions_registry.py`, which this runner reuses
for scoring (no new verifiers).

## Usage

```bash
CUDA_VISIBLE_DEVICES=6,7 \
python -m lm_eval.tasks.multi_if_fr \
    --model vllm \
    --model_args "pretrained=Qwen/Qwen3.5-0.8B,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.7,max_model_len=16384,enable_thinking=False" \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,max_gen_toks=1024 \
    --batch_size auto \
    --output_path eval_results/french_eval_result \
    --log_samples \
    --n_runs 10
```

- Generation defaults live in [`multi_if_fr.yaml`](multi_if_fr.yaml) and are overridden by
  `--gen_kwargs` (exactly like the harness's task `generation_kwargs`).
- `--n_runs > 1` writes each run to `run_{i}/<model>/` and an averaged `results_avg.json`,
  matching the `lm_eval --n_runs` layout so `scripts/make_table_results.py` can read it.
  Multiple runs are only meaningful with sampling (`do_sample=True`).
- `--limit N` restricts to the first N conversations (smoke tests).
- `--language` overrides the split (dataset stores full names: `French`, `Spanish`, …),
  so the same runner can evaluate other Multi-IF languages.

## Metrics

Four IFEval flavors are reported per turn (`turn_1` / `turn_2` / `turn_3`) plus a macro `avg`
across turns: `prompt_level_strict_acc`, `inst_level_strict_acc`, `prompt_level_loose_acc`,
`inst_level_loose_acc`.

**Headline: `multi_if_composite`.** Multi-IF is *not* reported like IFEval (prompt-strict).
Labs report a **composite = the mean of the four sub-metrics**, and the common convention
(Liquid AI LFM2 report, App. C: *"Average accuracy across all 3 turns"*; and the Multi-IF
paper's overall-ranking figure) is to **average that composite across the 3 turns**. That is
`multi_if_composite`. The alternative leaderboard/paper-analysis convention — the composite of
the **final turn only** — is emitted as `multi_if_composite_final_turn`.

Note: most labs (Qwen3, the llm-stats leaderboard) don't document their turn aggregation, so
cross-lab Multi-IF numbers are not perfectly comparable; `multi_if_composite` matches the one
lab (LFM2) that states it explicitly.
