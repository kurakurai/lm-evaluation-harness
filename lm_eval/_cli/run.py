import argparse
import json
import logging
import os
import textwrap
from copy import deepcopy
from functools import partial
from statistics import mean, stdev

from lm_eval._cli.subcommand import SubCommand
from lm_eval._cli.utils import (
    MergeDictAction,
    SplitArgs,
    _int_or_none_list_arg_type,
    request_caching_arg_to_dict,
    try_parse_json,
)


class Run(SubCommand):
    """Command for running language model evaluation."""

    def __init__(self, subparsers: argparse._SubParsersAction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parser = subparsers.add_parser(
            "run",
            help="Run the evaluation harness on specified tasks",
            description="Evaluate language models on various benchmarks and tasks.",
            usage="lm-eval run --model <model> --tasks <task> <task> --model_args <arg=value> <arg=value> [options]",
            epilog=textwrap.dedent("""
                examples:
                  # Basic evaluation with HuggingFace model
                  $ lm-eval run --model hf --model_args pretrained=gpt2 dtype=float32 --tasks hellaswag

                  # Evaluate on multiple tasks with few-shot examples
                  $ lm-eval run --model vllm --model_args pretrained=EleutherAI/gpt-j-6B --tasks arc_easy arc_challenge --num_fewshot 5

                  # Evaluation with custom generation parameters
                  $ lm-eval run --model hf --model_args pretrained=gpt2 --tasks lambada --gen_kwargs temperature=0.8 top_p=0.95 'stop=["\\n\\n"]'

                  # Use configuration file
                  $ lm-eval run --config my_config.yaml --tasks mmlu

                For more information, see: https://github.com/EleutherAI/lm-evaluation-harness
            """),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._add_args()
        self._parser.set_defaults(func=self._execute)

    def _add_args(self) -> None:
        self._parser = self._parser

        # Defaults are set in config/evaluate_config.py
        config_group = self._parser.add_argument_group("configuration")
        config_group.add_argument(
            "--config",
            "-C",
            default=None,
            type=str,
            metavar="<path>",
            help="Set initial arguments from YAML config",
        )

        # Model and Tasks
        model_group = self._parser.add_argument_group("model and tasks")
        model_group.add_argument(
            "--tasks",
            "-t",
            default=None,
            nargs="+",
            metavar="<task>",
            action=SplitArgs,
            help=textwrap.dedent("""
                Space (or comma-separated) list of task names or groupings.
                Use 'lm-eval list tasks' to see all available tasks.
            """).strip(),
        )
        model_group.add_argument(
            "--model",
            "-M",
            type=str,
            default=None,
            metavar="<model>",
            help="Model name (default: hf)",
        )
        model_group.add_argument(
            "--model_args",
            "-a",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<arg>",
            help="Model arguments as 'key=val,key2=val2' or `key=val` `key2=val2`",
        )
        model_group.add_argument(
            "--apply_chat_template",
            type=str,
            nargs="?",
            const=True,
            default=argparse.SUPPRESS,
            metavar="<template>",
            help="Apply chat template to prompts (optional template name)",
        )
        model_group.add_argument(
            "--limit",
            "-L",
            type=float,
            default=None,
            metavar="<limit>",
            help="Limit examples per task (integer count or fraction)",
        )
        model_group.add_argument(
            "--use_cache",
            "-c",
            type=str,
            default=None,
            metavar="<path>",
            help="Path to cache model responses (skips repeated inference)",
        )

        # Evaluation Settings
        eval_group = self._parser.add_argument_group("evaluation settings")
        eval_group.add_argument(
            "--n_runs",
            type=int,
            default=None,
            metavar="<n>",
            help="Number of times to run evaluation and average results (model loaded once).",
        )
        eval_group.add_argument(
            "--num_fewshot",
            "-f",
            type=int,
            default=None,
            metavar="<n>",
            help="Number of examples in few-shot context",
        )
        eval_group.add_argument(
            "--batch_size",
            "-b",
            type=str,
            default=argparse.SUPPRESS,
            metavar="<size>",
            help=textwrap.dedent(
                "Batch size: 'auto', 'auto:N' (auto-tune N times), or integer (default: 1)"
            ),
        )
        eval_group.add_argument(
            "--max_batch_size",
            type=int,
            default=None,
            metavar="<n>",
            help="Maximum batch size when using --batch_size auto",
        )
        eval_group.add_argument(
            "--device",
            type=str,
            default=None,
            metavar="<device>",
            help="Device to use (e.g. cuda, cuda:0, cpu, mps)",
        )
        eval_group.add_argument(
            "--gen_kwargs",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<arg>",
            help=textwrap.dedent(
                'Generation arguments as `temperature=0,stop=["stop"]` or `key=val` `key2=val2`.'
                "Values should be parsable with ast.literal_eval."
            ),
        )

        # Data and Output
        data_group = self._parser.add_argument_group(
            "data and output (see also: --limit)"
        )
        data_group.add_argument(
            "--output_path",
            "-o",
            default=None,
            type=str,
            metavar="<path>",
            help="Output dir or json file for results (and samples)",
        )
        data_group.add_argument(
            "--log_samples",
            "-s",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Save all model outputs and documents for post-hoc analysis",
        )
        data_group.add_argument(
            "--samples",
            "-E",
            default=None,
            type=try_parse_json,
            metavar="<json>",
            help="JSON mapping task names to sample indices, e.g. '{\"task1\": [0,1,2]}'. Incompatible with --limit.",
        )

        # Caching and Performance
        cache_group = self._parser.add_argument_group(
            "caching and performance (see also: --use_cache)"
        )
        cache_group.add_argument(
            "--cache_requests",
            type=request_caching_arg_to_dict,
            nargs="?",
            const="true",
            default=None,
            metavar="true|refresh|delete",
            help="Cache preprocessed prompts; bare flag defaults to 'true'",
        )
        cache_group.add_argument(
            "--check_integrity",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Run task test suite validation",
        )

        # Prompt Formatting
        template_group = self._parser.add_argument_group(
            "instruct formatting (see also: --apply_chat_template)"
        )
        template_group.add_argument(
            "--system_instruction",
            type=str,
            default=None,
            metavar="<text>",
            help="Add custom system instruction.",
        )
        template_group.add_argument(
            "--fewshot_as_multiturn",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            nargs="?",
            const=True,
            default=argparse.SUPPRESS,
            metavar="<bool>",
            help="Use fewshot as multi-turn conversation. Auto-enabled with --apply_chat_template. Use 'false' to disable.",
        )

        # Task Management
        task_group = self._parser.add_argument_group("task management")
        task_group.add_argument(
            "--include_path",
            type=str,
            default=None,
            metavar="<path>",
            help="Additional directory for external tasks",
        )

        # Logging and Tracking
        logging_group = self._parser.add_argument_group("logging and tracking")
        logging_group.add_argument(
            "--verbosity",
            "-v",
            type=str.upper,
            default=None,
            metavar="<level>",
            help="(Deprecated) Log level. Use LMEVAL_LOG_LEVEL env var instead",
        )
        logging_group.add_argument(
            "--write_out",
            "-w",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Print prompts for first few documents",
        )
        logging_group.add_argument(
            "--show_config",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Display full task configuration after evaluation",
        )
        logging_group.add_argument(
            "--wandb_args",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<args>",
            help="Weights & Biases init arguments key=val key2=val2",
        )
        logging_group.add_argument(
            "--wandb_config_args",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<args>",
            help="Weights & Biases config arguments key=val key2=val2",
        )
        logging_group.add_argument(
            "--trackio_args",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<args>",
            help="Trackio init arguments key=val key2=val2 (e.g. project=my-evals)",
        )
        logging_group.add_argument(
            "--hf_hub_log_args",
            default=None,
            nargs="+",
            action=MergeDictAction,
            metavar="<args>",
            help="Hugging Face Hub logging arguments key=val key2=val2",
        )

        # Advanced Options
        advanced_group = self._parser.add_argument_group("advanced options")
        advanced_group.add_argument(
            "--predict_only",
            "-x",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Save predictions only, skip metric computation",
        )
        default_seed_string = "0,1234,1234,1234"
        advanced_group.add_argument(
            "--seed",
            type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
            default=None,
            metavar="<seed>",
            help=textwrap.dedent(f"""
                Random seeds for python,numpy,torch,fewshot (default: {default_seed_string}).
                Use single integer for all, or comma-separated list of 4 values.
                Use 'None' to skip setting a seed. Example: --seed 42 or --seed 0,None,8,52
            """).strip(),
        )
        advanced_group.add_argument(
            "--trust_remote_code",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Allow executing remote code from Hugging Face Hub",
        )
        advanced_group.add_argument(
            "--confirm_run_unsafe_code",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Confirm understanding of unsafe code execution risks",
        )
        advanced_group.add_argument(
            "--metadata",
            type=json.loads,
            default=None,
            metavar="<arg>",
            help=textwrap.dedent(
                """`key=val` `key2=val` args parsable by ast.literal_eval (merged with model_args),
                required for some tasks such as RULER"""
            ),
        )

    @staticmethod
    def _execute(args: argparse.Namespace) -> None:
        """Runs the evaluation harness with the provided arguments."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        from lm_eval.config.evaluate_config import EvaluatorConfig

        eval_logger = logging.getLogger(__name__)

        # Create and validate config (most validation now occurs in EvaluationConfig)
        cfg = EvaluatorConfig.from_cli(args)

        import lm_eval.api.registry
        from lm_eval import simple_evaluate
        from lm_eval.loggers import EvaluationTracker, TrackioLogger, WandbLogger
        from lm_eval.utils import handle_non_serializable, make_table

        # Set up logging
        if cfg.wandb_args:
            wandb_logger = WandbLogger(cfg.wandb_args, cfg.wandb_config_args)
        if cfg.trackio_args:
            trackio_logger = TrackioLogger(cfg.trackio_args)

        # Log task selection (tasks already processed in config)
        if cfg.include_path is not None:
            eval_logger.info("Including path: %s", cfg.include_path)

        # Create task manager (metadata already set up in config validation)
        task_manager = cfg.process_tasks(cfg.metadata)

        eval_logger.info("Selected Tasks: %s", cfg.tasks)

        n_runs = cfg.n_runs if cfg.n_runs and cfg.n_runs > 1 else 1

        if n_runs > 1:
            # Initialize model once for all runs
            eval_logger.info(
                "Initializing model once for %d runs: %s", n_runs, cfg.model
            )
            lm = lm_eval.api.registry.get_model(cfg.model).create_from_arg_obj(
                cfg.model_args,
                {
                    "batch_size": cfg.batch_size,
                    "max_batch_size": cfg.max_batch_size,
                    "device": cfg.device,
                },
            )
            model_arg = lm
        else:
            model_arg = cfg.model

        def _run_single(run_idx: int, output_path: str | None) -> dict:
            """Run a single evaluation and return results."""
            # Set up evaluation tracker for this run
            run_hf_hub_log_args = dict(cfg.hf_hub_log_args)
            if output_path:
                run_hf_hub_log_args["output_path"] = output_path
            if os.environ.get("HF_TOKEN", None):
                run_hf_hub_log_args["token"] = os.environ.get("HF_TOKEN")
            run_tracker = EvaluationTracker(**run_hf_hub_log_args)

            # Vary seeds per run for statistical independence
            seeds = cfg.seed if cfg.seed else [None, None, None, None]
            run_results = simple_evaluate(
                model=model_arg,
                model_args=cfg.model_args,
                tasks=cfg.tasks,
                num_fewshot=cfg.num_fewshot,
                batch_size=cfg.batch_size,
                max_batch_size=cfg.max_batch_size,
                device=cfg.device,
                use_cache=cfg.use_cache,
                cache_requests=cfg.cache_requests.get("cache_requests", False),
                rewrite_requests_cache=cfg.cache_requests.get(
                    "rewrite_requests_cache", False
                ),
                delete_requests_cache=cfg.cache_requests.get(
                    "delete_requests_cache", False
                ),
                limit=cfg.limit,
                samples=cfg.samples,
                check_integrity=cfg.check_integrity,
                write_out=cfg.write_out,
                log_samples=cfg.log_samples,
                evaluation_tracker=run_tracker,
                system_instruction=cfg.system_instruction,
                apply_chat_template=cfg.apply_chat_template,
                fewshot_as_multiturn=cfg.fewshot_as_multiturn,
                gen_kwargs=cfg.gen_kwargs,
                task_manager=task_manager,
                verbosity=cfg.verbosity,
                predict_only=cfg.predict_only,
                random_seed=seeds[0] + run_idx if seeds[0] is not None else None,
                numpy_random_seed=seeds[1] + run_idx if seeds[1] is not None else None,
                torch_random_seed=seeds[2] + run_idx if seeds[2] is not None else None,
                fewshot_random_seed=seeds[3] + run_idx if seeds[3] is not None else None,
                confirm_run_unsafe_code=cfg.confirm_run_unsafe_code,
                metadata=cfg.metadata,
            )

            if run_results is not None:
                run_samples = None
                if cfg.log_samples:
                    run_samples = run_results.pop("samples")

                run_tracker.save_results_aggregated(
                    results=run_results, samples=run_samples if cfg.log_samples else None
                )

                if cfg.log_samples:
                    for task_name in run_results["configs"]:
                        run_tracker.save_results_samples(
                            task_name=task_name, samples=run_samples[task_name]
                        )

                if run_samples is not None:
                    run_results["samples"] = run_samples

            return run_results

        def _average_results(all_results: list) -> dict:
            """Average numeric metrics across runs.

            For n > 1 runs:
            - Value (e.g. ``acc,none``): mean across runs.
            - Stderr (e.g. ``acc_stderr,none``): REPLACED with the sample std
              (n-1) of the corresponding metric across runs. This is the
              cross-run variability — the conventional uncertainty for
              multi-seed evaluation, and what users typically expect from
              the ``±`` column.
            - ``_std`` fields (e.g. ``acc_std,none``): also kept explicitly,
              equal to the new stderr, for back-compat with downstream code.
            """
            avg = deepcopy(all_results[0])
            n = len(all_results)
            for section in ("results", "groups"):
                for task in avg.get(section, {}):
                    original_keys = list(avg[section][task].keys())
                    # First pass: replace every numeric field with its mean.
                    for key in original_keys:
                        val = avg[section][task][key]
                        if not isinstance(val, (int, float)):
                            continue
                        vals = [r[section][task].get(key, val) for r in all_results]
                        avg[section][task][key] = mean(vals)
                    if n <= 1:
                        continue
                    # Second pass: rewrite stderr fields as the cross-run
                    # sample std (n-1) of the corresponding metric, and add
                    # a parallel ``_std`` field for explicit access.
                    for key in original_keys:
                        val = avg[section][task][key]
                        if not isinstance(val, (int, float)):
                            continue
                        is_stderr = ("_stderr," in key) or key.endswith("_stderr")
                        if is_stderr:
                            if "_stderr," in key:
                                metric_key = key.replace("_stderr,", ",")
                            else:
                                metric_key = key[: -len("_stderr")]
                            if metric_key in avg[section][task]:
                                metric_vals = [
                                    r[section][task].get(metric_key, 0)
                                    for r in all_results
                                ]
                                avg[section][task][key] = stdev(metric_vals)
                        else:
                            std_key = (
                                key.replace(",", "_std,")
                                if "," in key
                                else key + "_std"
                            )
                            vals = [
                                r[section][task].get(key, val) for r in all_results
                            ]
                            avg[section][task][std_key] = stdev(vals)
            return avg

        # --- Main execution path ---
        if n_runs == 1:
            # Standard single-run path (unchanged behaviour)
            hf_hub_log_args = dict(cfg.hf_hub_log_args)
            if cfg.output_path:
                hf_hub_log_args["output_path"] = cfg.output_path
            if os.environ.get("HF_TOKEN", None):
                hf_hub_log_args["token"] = os.environ.get("HF_TOKEN")
            evaluation_tracker = EvaluationTracker(**hf_hub_log_args)

            if "push_samples_to_hub" in hf_hub_log_args and not cfg.log_samples:
                eval_logger.warning(
                    "Pushing samples to the Hub requires --log_samples to be set."
                )

            results = simple_evaluate(
                model=model_arg,
                model_args=cfg.model_args,
                tasks=cfg.tasks,
                num_fewshot=cfg.num_fewshot,
                batch_size=cfg.batch_size,
                max_batch_size=cfg.max_batch_size,
                device=cfg.device,
                use_cache=cfg.use_cache,
                cache_requests=cfg.cache_requests.get("cache_requests", False),
                rewrite_requests_cache=cfg.cache_requests.get(
                    "rewrite_requests_cache", False
                ),
                delete_requests_cache=cfg.cache_requests.get(
                    "delete_requests_cache", False
                ),
                limit=cfg.limit,
                samples=cfg.samples,
                check_integrity=cfg.check_integrity,
                write_out=cfg.write_out,
                log_samples=cfg.log_samples,
                evaluation_tracker=evaluation_tracker,
                system_instruction=cfg.system_instruction,
                apply_chat_template=cfg.apply_chat_template,
                fewshot_as_multiturn=cfg.fewshot_as_multiturn,
                gen_kwargs=cfg.gen_kwargs,
                task_manager=task_manager,
                verbosity=cfg.verbosity,
                predict_only=cfg.predict_only,
                random_seed=cfg.seed[0] if cfg.seed else None,
                numpy_random_seed=cfg.seed[1] if cfg.seed else None,
                torch_random_seed=cfg.seed[2] if cfg.seed else None,
                fewshot_random_seed=cfg.seed[3] if cfg.seed else None,
                confirm_run_unsafe_code=cfg.confirm_run_unsafe_code,
                metadata=cfg.metadata,
            )

            # Process results
            if results is not None:
                if cfg.log_samples:
                    samples = results.pop("samples")

                dumped = json.dumps(
                    results, indent=2, default=handle_non_serializable, ensure_ascii=False
                )
                if cfg.show_config:
                    print(dumped)

                batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

                # W&B logging
                if cfg.wandb_args:
                    try:
                        wandb_logger.post_init(results)
                        wandb_logger.log_eval_result()
                        if cfg.log_samples:
                            wandb_logger.log_eval_samples(samples)
                    except Exception as e:  # noqa: BLE001
                        eval_logger.info("Logging to W&B failed: %s", e)

                # Trackio logging
                if cfg.trackio_args:
                    try:
                        trackio_logger.post_init(results)
                        trackio_logger.log_eval_result()
                        if cfg.log_samples:
                            trackio_logger.log_eval_samples(samples)
                    except Exception as e:  # noqa: BLE001
                        eval_logger.info("Logging to Trackio failed: %s", e)

                # Save results
                evaluation_tracker.save_results_aggregated(
                    results=results, samples=samples if cfg.log_samples else None
                )

                if cfg.log_samples:
                    for task_name in results["configs"]:
                        evaluation_tracker.save_results_samples(
                            task_name=task_name, samples=samples[task_name]
                        )

                if (
                    evaluation_tracker.push_results_to_hub
                    or evaluation_tracker.push_samples_to_hub
                ):
                    evaluation_tracker.recreate_metadata_card()

                # Print results
                cfg.model_args.pop("trust_remote_code", None)
                print(
                    f"{cfg.model} ({cfg.model_args}), gen_kwargs: ({cfg.gen_kwargs}), "
                    f"limit: {cfg.limit}, num_fewshot: {cfg.num_fewshot}, "
                    f"batch_size: {cfg.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
                )
                print(make_table(results))
                if "groups" in results:
                    print(make_table(results, "groups"))

                if cfg.wandb_args:
                    wandb_logger.run.finish()

                if cfg.trackio_args:
                    trackio_logger.finish()

        else:
            # Multi-run path: run N times, save per-run results, then average
            if "push_samples_to_hub" in cfg.hf_hub_log_args and not cfg.log_samples:
                eval_logger.warning(
                    "Pushing samples to the Hub requires --log_samples to be set."
                )

            all_results = []
            for run_idx in range(n_runs):
                eval_logger.info("Starting run %d/%d", run_idx + 1, n_runs)
                run_output_path = (
                    os.path.join(cfg.output_path, f"run_{run_idx}")
                    if cfg.output_path
                    else None
                )
                run_results = _run_single(run_idx, run_output_path)
                if run_results is not None:
                    # Strip samples before averaging (already saved per-run)
                    run_results.pop("samples", None)
                    all_results.append(run_results)

            if all_results:
                results = _average_results(all_results)

                batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

                # Save averaged results
                if cfg.output_path:
                    avg_path = os.path.join(cfg.output_path, "results_avg.json")
                    os.makedirs(cfg.output_path, exist_ok=True)
                    with open(avg_path, "w", encoding="utf-8") as f:
                        json.dump(
                            results,
                            f,
                            indent=2,
                            default=handle_non_serializable,
                            ensure_ascii=False,
                        )
                    eval_logger.info("Averaged results saved to %s", avg_path)

                if cfg.show_config:
                    print(
                        json.dumps(
                            results,
                            indent=2,
                            default=handle_non_serializable,
                            ensure_ascii=False,
                        )
                    )

                # W&B logging (averaged results only)
                if cfg.wandb_args:
                    try:
                        wandb_logger.post_init(results)
                        wandb_logger.log_eval_result()
                    except Exception as e:  # noqa: BLE001
                        eval_logger.info("Logging to W&B failed: %s", e)

                # Trackio logging (averaged results only)
                if cfg.trackio_args:
                    try:
                        trackio_logger.post_init(results)
                        trackio_logger.log_eval_result()
                    except Exception as e:  # noqa: BLE001
                        eval_logger.info("Logging to Trackio failed: %s", e)

                cfg.model_args.pop("trust_remote_code", None)
                print(
                    f"{cfg.model} ({cfg.model_args}), gen_kwargs: ({cfg.gen_kwargs}), "
                    f"limit: {cfg.limit}, num_fewshot: {cfg.num_fewshot}, "
                    f"batch_size: {cfg.batch_size}{f' ({batch_sizes})' if batch_sizes else ''} "
                    f"[averaged over {n_runs} runs]"
                )
                print(make_table(results))
                if "groups" in results:
                    print(make_table(results, "groups"))

                if cfg.wandb_args:
                    wandb_logger.run.finish()

                if cfg.trackio_args:
                    trackio_logger.finish()
