"""Faithful multi-turn runner for Multi-IF (French split).

Multi-IF (https://huggingface.co/datasets/facebook/Multi-IF, arXiv:2410.15553)
extends IFEval to 3-turn conversations. The defining feature — and the reason it
cannot be a plain ``lm_eval --tasks`` entry — is that the model's own response to
turn N is fed back as context for turn N+1. The harness gathers every request for
a task and fires them at the model in a single batched call, so it has no way to
build turn N+1 from a response it has not produced yet.

This runner reuses the harness's own machinery so the launch experience matches the
other French tasks:

  * the model is built with the standard registry (``--model vllm --model_args ...``),
  * generation settings come from ``multi_if_fr.yaml`` and/or ``--gen_kwargs``,
  * scoring reuses the IFEval verifiers in ``lm_eval/tasks/ifeval`` (verified to
    cover every instruction id present in the French split),
  * results/samples are written in the same layout as ``lm_eval --n_runs``.

Example
-------
    python -m lm_eval.tasks.multi_if_fr \
        --model vllm \
        --model_args "pretrained=Qwen/Qwen3.5-0.8B,dtype=bfloat16,tensor_parallel_size=2,gpu_memory_utilization=0.7,max_model_len=16384,enable_thinking=False" \
        --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,max_gen_toks=1024 \
        --batch_size auto \
        --output_path eval_results/french_eval_result \
        --log_samples \
        --n_runs 10
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from statistics import mean, stdev

import yaml
from datasets import load_dataset


eval_logger = logging.getLogger("multi_if_fr")

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "multi_if_fr.yaml")
_METRICS = (
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m lm_eval.tasks.multi_if_fr",
        description="Faithful multi-turn Multi-IF (French) evaluation.",
    )
    p.add_argument("--model", required=True, help="Model type, e.g. 'vllm' or 'hf'.")
    p.add_argument(
        "--model_args",
        default="",
        help="Comma-separated model args, e.g. 'pretrained=...,tensor_parallel_size=2'.",
    )
    p.add_argument(
        "--gen_kwargs",
        default="",
        help="Comma-separated generation args overriding the YAML defaults, "
        "e.g. 'do_sample=True,temperature=0.6,max_gen_toks=1024'.",
    )
    p.add_argument("--batch_size", default="auto")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Path to the task YAML (dataset + default generation settings).",
    )
    p.add_argument(
        "--language",
        default=None,
        help="Override the dataset 'language' column value to keep (default: from YAML).",
    )
    p.add_argument("--output_path", default=None, help="Directory to write results into.")
    p.add_argument("--log_samples", action="store_true", help="Write a per-conversation JSONL.")
    p.add_argument("--limit", type=int, default=None, help="Only evaluate the first N conversations.")
    p.add_argument("--n_runs", type=int, default=1, help="Number of runs to average (use with sampling).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--verbosity", default="INFO")
    return p.parse_args(argv)


def _sanitize(name: str) -> str:
    return name.replace("/", "__")


def _model_name_from_args(model_args: str) -> str:
    for part in model_args.split(","):
        if part.strip().startswith("pretrained="):
            return part.split("=", 1)[1].strip()
    return "model"


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _turn_prompt_content(raw) -> str:
    """turn_N_prompt is stored as a JSON string {"role": ..., "content": ...}."""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    return obj["content"]


def _parse_kwargs_list(raw):
    """turn_N_kwargs is a list of JSON strings (one dict per instruction)."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    out = []
    for item in raw:
        out.append(json.loads(item) if isinstance(item, str) else item)
    return out


def _load_conversations(config: dict, language: str, limit):
    ds = load_dataset(config["dataset_path"], split=config.get("dataset_split", "train"))
    ds = ds.filter(lambda x: x["language"] == language)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    convs = []
    max_turns = int(config.get("num_turns", 3))
    # A minority of conversations have fewer than max_turns (later turn columns
    # are None); keep only the turns that are actually present.
    for row in ds:
        turns = []
        for t in range(1, max_turns + 1):
            if row[f"turn_{t}_prompt"] is None:
                break
            turns.append(
                {
                    "prompt": _turn_prompt_content(row[f"turn_{t}_prompt"]),
                    "instruction_id_list": (
                        json.loads(row[f"turn_{t}_instruction_id_list"])
                        if isinstance(row[f"turn_{t}_instruction_id_list"], str)
                        else row[f"turn_{t}_instruction_id_list"]
                    ),
                    "kwargs": _parse_kwargs_list(row[f"turn_{t}_kwargs"]),
                }
            )
        convs.append({"key": row["key"], "turns": turns})
    return convs, max_turns


def _score_turn(prompt, instruction_id_list, kwargs, response):
    """Score one response against the (cumulative) instruction list for its turn.

    Reuses the standard IFEval verifiers; kwargs are self-contained in Multi-IF
    (e.g. 'prompt_to_repeat' is embedded), so no prompt injection is needed beyond
    what the shared helpers already do.
    """
    from lm_eval.tasks.ifeval.utils import (
        InputExample,
        test_instruction_following_loose,
        test_instruction_following_strict,
    )

    inp = InputExample(
        key=0,
        instruction_id_list=instruction_id_list,
        prompt=prompt,
        kwargs=kwargs,
    )
    strict = test_instruction_following_strict(inp, response)
    loose = test_instruction_following_loose(inp, response)
    return {
        "prompt_level_strict_acc": strict.follow_all_instructions,
        "inst_level_strict_acc": strict.follow_instruction_list,
        "prompt_level_loose_acc": loose.follow_all_instructions,
        "inst_level_loose_acc": loose.follow_instruction_list,
    }


def _build_model(args):
    import lm_eval.api.registry

    model_cls = lm_eval.api.registry.get_model(args.model)
    return model_cls.create_from_arg_string(
        args.model_args,
        {"batch_size": args.batch_size, "max_batch_size": None, "device": args.device},
    )


def _generate_turn(lm, contexts, gen_kwargs):
    """Batched generate_until over one turn's chat-templated contexts."""
    from lm_eval.api.instance import Instance

    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(ctx, dict(gen_kwargs)),
            idx=i,
        )
        for i, ctx in enumerate(contexts)
    ]
    return lm.generate_until(requests)


def _run_once(lm, convs, num_turns, gen_kwargs):
    """Run the full multi-turn loop; return (per_turn_metrics, per_conv_samples).

    Turn count is variable across conversations, so each turn only generates for
    (and scores) the conversations that actually have that turn.
    """
    histories = [[] for _ in convs]
    # collected[t] = list of score dicts for the conversations present at turn t.
    collected = {t: [] for t in range(1, num_turns + 1)}
    # per-conversation ordered list of (response, score) for sample logging.
    per_conv = [[] for _ in convs]

    for t in range(1, num_turns + 1):
        participants = [i for i, c in enumerate(convs) if len(c["turns"]) >= t]
        if not participants:
            continue
        contexts = []
        for i in participants:
            turn = convs[i]["turns"][t - 1]
            histories[i].append({"role": "user", "content": turn["prompt"]})
            contexts.append(lm.apply_chat_template(histories[i]))

        eval_logger.info(
            "Generating turn %d/%d (%d conversations)", t, num_turns, len(participants)
        )
        outputs = _generate_turn(lm, contexts, gen_kwargs)

        for idx, i in enumerate(participants):
            resp = outputs[idx]
            histories[i].append({"role": "assistant", "content": resp})
            turn = convs[i]["turns"][t - 1]
            score = _score_turn(
                turn["prompt"], turn["instruction_id_list"], turn["kwargs"], resp
            )
            collected[t].append(score)
            per_conv[i].append((resp, score))

    per_turn = _aggregate_turns(collected, num_turns)
    samples = _build_samples(convs, per_conv)
    return per_turn, samples


def _aggregate_turns(collected, num_turns):
    per_turn = {}
    present_turns = []
    for t in range(1, num_turns + 1):
        scores = collected[t]
        if scores:
            present_turns.append(t)
        for metric in _METRICS:
            if metric.startswith("inst_"):
                flat = [x for s in scores for x in s[metric]]
                val = sum(flat) / len(flat) if flat else 0.0
            else:
                val = mean(bool(s[metric]) for s in scores) if scores else 0.0
            per_turn[f"turn_{t}_{metric}"] = val
    # Macro average across the turns that actually have data.
    for metric in _METRICS:
        per_turn[f"avg_{metric}"] = (
            mean(per_turn[f"turn_{t}_{metric}"] for t in present_turns)
            if present_turns
            else 0.0
        )
    # Composite = mean of the 4 sub-metrics (the convention labs report). Computed
    # per turn, plus the headline aggregation choices.
    for t in present_turns:
        per_turn[f"turn_{t}_composite"] = mean(
            per_turn[f"turn_{t}_{m}"] for m in _METRICS
        )
    # Headline: composite averaged across all turns (LFM2 App. C convention, and
    # the paper's overall-ranking figure). `..._final_turn` is the alternative
    # leaderboard/paper-analysis convention (turn 3 only), kept for reference.
    per_turn["multi_if_composite"] = (
        mean(per_turn[f"turn_{t}_composite"] for t in present_turns)
        if present_turns
        else 0.0
    )
    per_turn["multi_if_composite_final_turn"] = (
        per_turn[f"turn_{present_turns[-1]}_composite"] if present_turns else 0.0
    )
    return per_turn


def _build_samples(convs, per_conv):
    samples = []
    for i, conv in enumerate(convs):
        turns = []
        for t, (resp, score) in enumerate(per_conv[i], start=1):
            turn = conv["turns"][t - 1]
            turns.append(
                {
                    "turn": t,
                    "prompt": turn["prompt"],
                    "instruction_id_list": turn["instruction_id_list"],
                    "response": resp,
                    "prompt_level_strict_acc": bool(score["prompt_level_strict_acc"]),
                    "inst_level_strict_acc": score["inst_level_strict_acc"],
                    "prompt_level_loose_acc": bool(score["prompt_level_loose_acc"]),
                    "inst_level_loose_acc": score["inst_level_loose_acc"],
                }
            )
        samples.append({"key": conv["key"], "num_turns": len(turns), "turns": turns})
    return samples


def _results_dict(per_turn, config, model_name, gen_kwargs, n):
    metrics = {}
    for k, v in per_turn.items():
        metrics[f"{k},none"] = v
        metrics[f"{k}_stderr,none"] = "N/A"
    version = float(config.get("metadata", {}).get("version", 1.0))
    return {
        "results": {"multi_if_fr": {"alias": "multi_if_fr", **metrics}},
        "versions": {"multi_if_fr": version},
        "n-samples": {"multi_if_fr": {"original": n, "effective": n}},
        "config": {
            "model": model_name,
            "gen_kwargs": gen_kwargs,
            "dataset_path": config["dataset_path"],
            "language": config.get("language"),
            "num_turns": config.get("num_turns", 3),
        },
        "configs": {"multi_if_fr": config},
    }


def _write_run(output_dir, model_name, results, samples, date_id, log_samples):
    model_dir = os.path.join(output_dir, _sanitize(model_name))
    os.makedirs(model_dir, exist_ok=True)
    results_path = os.path.join(model_dir, f"results_{date_id}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    eval_logger.info("Wrote %s", results_path)
    if log_samples and samples is not None:
        samples_path = os.path.join(model_dir, f"samples_multi_if_fr_{date_id}.jsonl")
        with open(samples_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        eval_logger.info("Wrote %s", samples_path)


def _average(all_per_turn):
    keys = all_per_turn[0].keys()
    avg = {}
    for k in keys:
        vals = [r[k] for r in all_per_turn]
        avg[k] = mean(vals)
        if len(vals) > 1:
            avg[f"{k}_std"] = stdev(vals)
    return avg


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.verbosity.upper(), logging.INFO))

    from lm_eval.utils import simple_parse_args_string

    config = _load_config(args.config)
    language = args.language or config.get("language", "French")

    gen_kwargs = dict(config.get("generation_kwargs", {}) or {})
    gen_kwargs.update(simple_parse_args_string(args.gen_kwargs))
    gen_kwargs.setdefault("until", [])

    model_name = _model_name_from_args(args.model_args)
    convs, num_turns = _load_conversations(config, language, args.limit)
    eval_logger.info(
        "Loaded %d %s conversations (%d turns each).", len(convs), language, num_turns
    )

    lm = _build_model(args)

    n_runs = max(1, args.n_runs)
    all_per_turn = []
    for run_idx in range(n_runs):
        date_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
        eval_logger.info("=== Run %d/%d ===", run_idx + 1, n_runs)
        per_turn, samples = _run_once(lm, convs, num_turns, gen_kwargs)
        all_per_turn.append(per_turn)

        results = _results_dict(per_turn, config, model_name, gen_kwargs, len(convs))
        if args.output_path:
            run_dir = (
                os.path.join(args.output_path, f"run_{run_idx}")
                if n_runs > 1
                else args.output_path
            )
            _write_run(run_dir, model_name, results, samples, date_id, args.log_samples)

    final = all_per_turn[0] if n_runs == 1 else _average(all_per_turn)
    if args.output_path and n_runs > 1:
        os.makedirs(args.output_path, exist_ok=True)
        avg_results = _results_dict(final, config, model_name, gen_kwargs, len(convs))
        with open(os.path.join(args.output_path, "results_avg.json"), "w", encoding="utf-8") as f:
            json.dump(avg_results, f, indent=2, ensure_ascii=False)

    _print_summary(final, num_turns, n_runs)
    return final


def _print_summary(final, num_turns, n_runs):
    label_w, col_w = 26, 9
    columns = [f"turn_{t}" for t in range(1, num_turns + 1)] + ["avg"]
    width = label_w + col_w * len(columns)
    print("\n" + "=" * width)
    print(f"Multi-IF (French) results  (n_runs={n_runs})")
    print("=" * width)
    print(f"{'metric':<{label_w}}" + "".join(f"{c:>{col_w}}" for c in columns))
    for metric in _METRICS:
        keys = [f"turn_{t}_{metric}" for t in range(1, num_turns + 1)] + [f"avg_{metric}"]
        row = f"{metric:<{label_w}}" + "".join(f"{final[k] * 100:>{col_w}.2f}" for k in keys)
        print(row)
    # Composite row (mean of the 4 sub-metrics), per turn + avg-over-turns.
    comp_keys = [f"turn_{t}_composite" for t in range(1, num_turns + 1)] + ["multi_if_composite"]
    print("-" * width)
    print(
        f"{'composite (mean of 4)':<{label_w}}"
        + "".join(f"{final[k] * 100:>{col_w}.2f}" for k in comp_keys)
    )
    print("=" * width)
    print(
        f"HEADLINE  multi_if_composite (avg over turns) = {final['multi_if_composite'] * 100:.2f}"
        f"   |  final-turn = {final['multi_if_composite_final_turn'] * 100:.2f}"
    )
    print("=" * width)


if __name__ == "__main__":
    main()
