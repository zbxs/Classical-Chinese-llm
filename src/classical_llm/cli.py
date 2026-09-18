from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="classical-llm", description="Classical Chinese LLM lab")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect the isolated environment and CUDA")

    reuse = commands.add_parser("audit-resources", help="Verify reusable local resources")
    reuse.add_argument("--config", default="configs/local_resources.yaml")

    acquire = commands.add_parser("acquire", help="Stream enabled Hugging Face sources")
    acquire.add_argument("--config", default="configs/data/sources.yaml")
    acquire.add_argument("--output", default="data/raw/huggingface")
    acquire.add_argument("--max-documents", type=int)

    inspect_hf = commands.add_parser("inspect-hf", help="Read one row to inspect a source schema")
    inspect_hf.add_argument("--dataset", required=True)
    inspect_hf.add_argument("--config")
    inspect_hf.add_argument("--output")

    prepare = commands.add_parser("prepare", help="Clean, deduplicate and split JSONL corpora")
    prepare.add_argument("inputs", nargs="+")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=20260912)

    mix = commands.add_parser("mix", help="Build a token-ratio pretraining mixture")
    mix.add_argument("--input", required=True)
    mix.add_argument("--output", required=True)
    mix.add_argument("--tokenizer", required=True)
    mix.add_argument("--total-tokens", required=True, type=int)
    mix.add_argument("--fractions", required=True, help='JSON, e.g. {"classical":0.2,"general_zh":0.8}')

    sft = commands.add_parser("build-sft", help="Build grounded conversational SFT data")
    sft.add_argument("inputs", nargs="+")
    sft.add_argument("--output", required=True)
    sft.add_argument("--quotas", help="JSON task-to-count mapping")

    teacher = commands.add_parser(
        "synthesize-sft", help="Generate auditable appreciation/creation SFT candidates"
    )
    teacher.add_argument("--input", required=True)
    teacher.add_argument("--output", required=True)
    teacher.add_argument("--teacher", required=True)
    teacher.add_argument("--tasks", default="appreciation,creation")
    teacher.add_argument("--limit", required=True, type=int)
    teacher.add_argument("--max-new-tokens", type=int, default=384)

    pref = commands.add_parser("build-preferences", help="Create labeled preference pairs")
    pref.add_argument("--input", required=True)
    pref.add_argument("--output", required=True)
    pref.add_argument("--limit", type=int, default=20000)

    ev = commands.add_parser("build-eval", help="Build the source-held-out 4x100 evaluation set")
    ev.add_argument("inputs", nargs="+")
    ev.add_argument("--output", required=True)
    ev.add_argument("--per-task", type=int, default=100)

    general_ev = commands.add_parser(
        "build-general-eval", help="Freeze held-out classical/Chinese/English perplexity sets"
    )
    general_ev.add_argument("--input", required=True)
    general_ev.add_argument("--output", required=True)
    general_ev.add_argument("--per-category", type=int, default=200)

    train = commands.add_parser("train", help="Run one registered training stage")
    train.add_argument("stage", choices=["cpt", "sft", "reward", "dpo", "grpo"])
    train.add_argument("--config", required=True)

    generate = commands.add_parser("generate", help="Generate evaluation responses")
    generate.add_argument("--model", required=True)
    generate.add_argument("--dataset", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=384)

    score = commands.add_parser("score", help="Compute deterministic automatic metrics")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)

    blind = commands.add_parser("blind-review", help="Create an anonymized human-review sheet")
    blind.add_argument("--generations", required=True, help="JSON model-name to generation-file map")
    blind.add_argument("--output", required=True)

    review = commands.add_parser("summarize-review", help="Aggregate completed anonymous 1-5 scores")
    review.add_argument("--input", required=True)
    review.add_argument("--key", required=True)
    review.add_argument("--output", required=True)

    perplexity = commands.add_parser(
        "perplexity", help="Measure token-weighted perplexity on a fixed general/domain corpus"
    )
    perplexity.add_argument("--model", required=True)
    perplexity.add_argument("--dataset", required=True)
    perplexity.add_argument("--output", required=True)
    perplexity.add_argument("--max-length", type=int, default=1024)
    perplexity.add_argument("--max-documents", type=int)

    report = commands.add_parser("report", help="Collect actual run and evaluation metadata")
    report.add_argument("--output", default="reports/generated/results.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    from classical_llm.runtime import configure_project_environment

    configure_project_environment()
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        from classical_llm.doctor import save_environment_report

        _emit(save_environment_report())
    elif args.command == "audit-resources":
        from classical_llm.resources import audit_resources

        _emit(audit_resources(args.config))
    elif args.command == "acquire":
        from classical_llm.data.acquire import acquire_huggingface_sources

        _emit(acquire_huggingface_sources(args.config, args.output, args.max_documents))
    elif args.command == "inspect-hf":
        from classical_llm.data.acquire import inspect_huggingface_source

        result = inspect_huggingface_source(args.dataset, args.config)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
        _emit(result)
    elif args.command == "prepare":
        from classical_llm.data.prepare import prepare_corpus

        _emit(prepare_corpus(args.inputs, args.output, args.seed))
    elif args.command == "mix":
        from classical_llm.data.mixture import build_token_mixture

        _emit(
            build_token_mixture(
                args.input, args.output, args.tokenizer, args.total_tokens, json.loads(args.fractions)
            )
        )
    elif args.command == "build-sft":
        from classical_llm.data.sft import build_sft_dataset

        quotas = json.loads(args.quotas) if args.quotas else None
        _emit(build_sft_dataset(args.inputs, args.output, quotas))
    elif args.command == "synthesize-sft":
        from classical_llm.data.teacher import synthesize_sft

        tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        _emit(
            synthesize_sft(
                args.input,
                args.output,
                args.teacher,
                tasks,
                args.limit,
                max_new_tokens=args.max_new_tokens,
            )
        )
    elif args.command == "build-preferences":
        from classical_llm.data.preferences import build_preferences

        _emit(build_preferences(args.input, args.output, args.limit))
    elif args.command == "build-eval":
        from classical_llm.data.evaluation import build_domain_evaluation

        _emit(build_domain_evaluation(args.inputs, args.output, args.per_task))
    elif args.command == "build-general-eval":
        from classical_llm.data.evaluation import build_perplexity_sets

        _emit(build_perplexity_sets(args.input, args.output, args.per_category))
    elif args.command == "train":
        runners = {
            "cpt": "classical_llm.training.cpt:run_cpt",
            "sft": "classical_llm.training.sft:run_sft",
            "reward": "classical_llm.training.reward:run_reward",
            "dpo": "classical_llm.training.dpo:run_dpo",
            "grpo": "classical_llm.training.grpo:run_grpo",
        }
        module_name, function_name = runners[args.stage].split(":")
        module = __import__(module_name, fromlist=[function_name])
        _emit(getattr(module, function_name)(args.config))
    elif args.command == "generate":
        from classical_llm.evaluation.generate import generate_responses

        _emit(generate_responses(args.model, args.dataset, args.output, args.max_new_tokens))
    elif args.command == "score":
        from classical_llm.evaluation.score import score_generation_file

        _emit(score_generation_file(args.input, args.output))
    elif args.command == "blind-review":
        from classical_llm.evaluation.score import make_blinded_review

        _emit(make_blinded_review(json.loads(args.generations), args.output))
    elif args.command == "summarize-review":
        from classical_llm.evaluation.score import summarize_blinded_review

        _emit(summarize_blinded_review(args.input, args.key, args.output))
    elif args.command == "perplexity":
        from classical_llm.evaluation.perplexity import evaluate_perplexity

        _emit(
            evaluate_perplexity(
                args.model,
                args.dataset,
                args.output,
                args.max_length,
                args.max_documents,
            )
        )
    elif args.command == "report":
        from classical_llm.reporting import collect_results

        _emit(collect_results(args.output))


if __name__ == "__main__":
    main()
