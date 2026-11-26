"""Command line entry point: ``semjoin <command>`` or ``python -m semjoin <command>``."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from semjoin.datasets import DATASETS, load_tasks
from semjoin.splits import load_manifest, write_manifest


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _matcher_list(value: str) -> list[str]:
    # Matcher specs contain commas inside their arguments ("fuzzy:threshold=0.7,q=2"),
    # so --matchers is separated by semicolons or given multiple times.
    return [v.strip() for v in value.split(";") if v.strip()]


def cmd_download_autofj(args: argparse.Namespace) -> int:
    from semjoin.datasets import autofj

    names = autofj.fetch(args.data_root, args.names)
    autofj.build(args.data_root, names=names)
    print(f"{len(names)} Auto-FuzzyJoin tasks in {args.data_root / 'autofj'}")
    return 0


def cmd_download_autojoin(args: argparse.Namespace) -> int:
    from semjoin.datasets import autojoin

    autojoin.fetch(args.data_root)
    built = autojoin.build(args.data_root)
    print(f"{len(built)} Auto-Join web-table tasks in {args.data_root / 'autojoin'} (held out: test only)")
    return 0


def cmd_download_nyc(args: argparse.Namespace) -> int:
    from semjoin.datasets import nyc

    nyc.fetch(args.data_root, refresh=args.refresh)
    built = nyc.build(args.data_root)
    print(f"{len(built)} NYC Open Data tasks in {args.data_root / 'nyc'} (held out: test only)")
    return 0


def cmd_build_mapping(args: argparse.Namespace) -> int:
    from semjoin.datasets import mapping, mixed

    out = args.data_root / "mapping"
    built = mapping.build(out, seed=args.seed, sample_fraction=args.sample_fraction)
    built += mixed.build(args.data_root, out, seed=args.seed)
    print(f"built {len(built)} mapping tasks in {out}: {', '.join(built)}")
    return 0


def cmd_download_wikidata(args: argparse.Namespace) -> int:
    from semjoin.datasets import wikidata

    names = wikidata.fetch(args.data_root, args.names, refresh=args.refresh)
    print(f"{len(names)} Wikidata tables cached in {args.data_root / 'raw' / 'wikidata'}")
    return 0


def cmd_build_wikidata(args: argparse.Namespace) -> int:
    from semjoin.datasets import wikidata

    built = wikidata.build(args.data_root, names=args.names)
    print(f"built {len(built)} wikidata tasks in {args.data_root / 'wikidata'}")
    return 0


def cmd_download_freyja(args: argparse.Namespace) -> int:
    from semjoin.datasets import freyja

    freyja.fetch(args.data_root)
    cands = freyja.candidates(args.data_root)
    out = args.data_root / "freyja_candidates.csv"
    freyja.write_candidates(cands, out)
    print(f"FREYJA lake cached in {args.data_root / 'raw' / 'freyja'}; "
          f"{len(cands)} joinable column pairs ranked for labeling in {out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from semjoin.validate import validate

    manifest = load_manifest(args.manifest) if args.manifest.exists() else None
    report = validate(args.data_root, args.datasets, manifest)
    print(report.format())
    return 0 if report.ok(strict=args.strict) else 1


def cmd_split(args: argparse.Namespace) -> int:
    tasks = load_tasks(list(DATASETS), args.data_root)
    assignment = write_manifest(args.manifest, tasks, seed=args.seed, dev_fraction=args.dev_fraction)
    counts: dict[str, dict[str, int]] = {}
    for key, split in assignment.items():
        ds = key.split("/", 1)[0]
        counts.setdefault(ds, {"dev": 0, "test": 0})[split] += 1
    for ds, c in sorted(counts.items()):
        print(f"{ds}: dev={c['dev']} test={c['test']}")
    print(f"wrote {args.manifest}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from semjoin.run import run

    manifest = load_manifest(args.manifest)
    split = None if args.split == "all" else args.split
    tasks = load_tasks(args.datasets, args.data_root, manifest, split)
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t.task_id in wanted or t.key in wanted]
    if args.max_tasks_per_type is not None:
        from semjoin.slicing import select_tasks

        tasks = select_tasks(tasks, args.max_tasks_per_type)
    if args.max_right is not None:
        from semjoin.slicing import slice_right

        tasks = [slice_right(t, args.max_right) for t in tasks]
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 1
    matchers = [m for group in args.matchers for m in group]

    if args.llm == "ollama":
        from semjoin.llm import OllamaClient

        llm_factory = lambda: OllamaClient(model=args.ollama_model, url=args.ollama_url)  # noqa: E731
    else:
        from semjoin.llm import FakeLLM

        llm_factory = FakeLLM

    out_dir = args.out or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    config = {
        "argv": sys.argv,
        "datasets": args.datasets,
        "split": args.split,
        "llm": args.llm,
        "ollama_model": args.ollama_model if args.llm == "ollama" else None,
        "slice": {"max_tasks_per_type": args.max_tasks_per_type, "max_right": args.max_right},
    }
    run(tasks, matchers, llm_factory, out_dir, config)
    print(f"results in {out_dir}")
    if not args.no_summary:
        from semjoin.summarize import summarize

        print(summarize(out_dir))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    from semjoin.summarize import summarize

    if args.router and not args.strategies:
        print("--router needs --strategies", file=sys.stderr)
        return 2
    strategies = [m for group in args.strategies or [] for m in group]
    print(summarize(args.run_dir, args.router, strategies))
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    from semjoin.tune import parse_grid, tune

    strategies = [m for group in args.strategies or [] for m in group] or None
    grid = parse_grid(args.grid) if args.grid else None
    print(tune(args.run_dir, args.router, strategies, grid, args.objective, args.top))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semjoin", description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--manifest", type=Path, default=Path("splits.json"))
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("download-autofj", help="download the Auto-FuzzyJoin benchmark")
    s.add_argument("--names", type=_csv, help="comma-separated subset of task names")
    s.set_defaults(func=cmd_download_autofj)

    s = sub.add_parser("download-autojoin", help="download the Auto-Join web tables (held-out real-world set)")
    s.set_defaults(func=cmd_download_autojoin)

    s = sub.add_parser("download-nyc", help="query NYC Open Data joins (held-out real-world set)")
    s.add_argument("--refresh", action="store_true", help="re-query tables already cached")
    s.set_defaults(func=cmd_download_nyc)

    s = sub.add_parser("build-mapping", help="generate the mapping-table dataset (incl. mixed-type tasks)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--sample-fraction", type=float, default=0.6)
    s.set_defaults(func=cmd_build_mapping)

    s = sub.add_parser("download-wikidata", help="re-query Wikidata (data/wikidata/ is committed; only to update it)")
    s.add_argument("--names", type=_csv, help="comma-separated subset of tables")
    s.add_argument("--refresh", action="store_true", help="re-query tables that are already cached")
    s.set_defaults(func=cmd_download_wikidata)

    s = sub.add_parser("build-wikidata", help="rebuild data/wikidata/ from the downloaded tables")
    s.add_argument("--names", type=_csv, help="comma-separated subset of tables")
    s.set_defaults(func=cmd_build_wikidata)

    s = sub.add_parser("download-freyja", help="download the FREYJA lake and rank column pairs to label")
    s.set_defaults(func=cmd_download_freyja)

    s = sub.add_parser("split", help="assign dev/test splits to new tasks in the manifest")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dev-fraction", type=float, default=0.3)
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("validate", help="check tasks, split coverage and downloads before pushing data")
    s.add_argument("--datasets", type=_csv, default=list(DATASETS))
    s.add_argument("--strict", action="store_true", help="fail on warnings too")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("run", help="run matchers on tasks")
    s.add_argument("--datasets", type=_csv, default=list(DATASETS))
    s.add_argument(
        "--matchers",
        type=_matcher_list,
        action="append",
        required=True,
        help='matcher specs separated by ";", e.g. "exact;fuzzy:threshold=0.7"',
    )
    s.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    s.add_argument("--tasks", type=_csv, help="restrict to these task ids or dataset/task keys")
    s.add_argument("--max-tasks-per-type", type=int, help="slice: at most N tasks per (dataset, mismatch type)")
    s.add_argument("--max-right", type=int, help="slice: at most N right values per task")
    s.add_argument("--llm", choices=["fake", "ollama"], default="fake")
    s.add_argument("--ollama-model", default=None, help="default: $SEMJOIN_OLLAMA_MODEL or qwen2.5:7b")
    s.add_argument("--ollama-url", default=None, help="default: $SEMJOIN_OLLAMA_URL or http://localhost:11434")
    s.add_argument("--out", type=Path, help="run directory (default: runs/<timestamp>)")
    s.add_argument("--no-summary", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize", help="aggregate a run directory")
    s.add_argument("run_dir", type=Path)
    s.add_argument("--router", help="matcher spec of the router to evaluate (RQ1)")
    s.add_argument("--strategies", type=_matcher_list, action="append", help="strategy specs the router chooses between")
    s.set_defaults(func=cmd_summarize)

    s = sub.add_parser("tune", help="score other router thresholds on a finished run, without re-running it")
    s.add_argument("run_dir", type=Path)
    s.add_argument("--router", default="router", help="the router spec as it was run (default: router)")
    s.add_argument("--strategies", type=_matcher_list, action="append", help="default: every strategy the router can pick")
    s.add_argument("--grid", help='thresholds to try, e.g. "typo_cos=0.4,0.5;code_len=3,4" (default: a built-in grid)')
    s.add_argument("--objective", choices=["macro_f1", "micro_f1", "accuracy", "regret"], default="macro_f1")
    s.add_argument("--top", type=int, default=10, help="how many of the best settings to list")
    s.set_defaults(func=cmd_tune)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
