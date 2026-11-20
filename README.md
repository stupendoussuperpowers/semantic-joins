# semantic-joins

Evaluation harness for semantic join strategies: dataset loading, dev/test
splits, quality and cost metrics, and result summaries.

## Setup

```sh
uv sync --extra dev --extra plot      # or: pip install -e '.[dev,plot]'
```

## Data

```sh
semjoin download-autofj   # 50 Auto-FuzzyJoin tasks -> data/autofj/
semjoin build-mapping     # ISO-based code/alias/typo/reorder/acronym tasks -> data/mapping/
semjoin split             # assign new tasks to dev/test in splits.json (commit it)
```

Freyja tasks are labeled by hand in `data/freyja/`; see
[docs/data-format.md](docs/data-format.md) for the format.
`splits.json` is append-only: rerunning `split` only assigns tasks that are
not in it yet.

## Running

```sh
semjoin run --split dev --matchers "exact;normalized;fuzzy:threshold=0.7"
semjoin run --split test --datasets autofj --matchers "fuzzy" --llm ollama \
    --ollama-model qwen2.5:7b --ollama-url http://gpu-box:11434
semjoin summarize runs/<id>
semjoin summarize runs/<id> --router router --strategies "fuzzy;llm"
```

`--matchers` takes specs separated by `;`; arguments after `:` are passed to
the matcher's constructor. `--llm fake` (the default) uses an offline stub.

Each run writes `runs/<timestamp>/`:

| file | contents |
|---|---|
| `config.json` | matchers, task list, LLM settings, git commit |
| `results.jsonl` | per (task, matcher): P/R/F1, hard-pair counts, LLM calls, tokens, runtime per stage, annotations |
| `errors.jsonl` | per (task, matcher): false positive and false negative pairs |
| `summary.md`, `summary.csv` | per (matcher, dataset): micro P/R/F1, macro F1, hard F1, LLM calls, tokens, runtime |
| `by_type.csv` | the same, per mismatch type |
| `routing.md` | with `--router`: routing accuracy and F1 regret vs. the best strategy per task |
| `cost_quality.png` | micro F1 vs. LLM calls per matcher |

**Hard F1** counts only pairs whose values differ after normalization, i.e.
the pairs a normalized equi-join cannot find.

## Matchers

Baselines: `exact`, `normalized`, `fuzzy`. Strategies: `autofj`, `transform`,
`acronym`, `llm_gen`, `llm`, `cascade`. Each strategy runs a normalized join, then
its own method, then (with `residual=true`) an LLM pass on what's left; see `semjoin/matchers/`.

Examples:
```sh
semjoin run --split dev --matchers "autofj"
semjoin run --split dev --matchers "acronym:residual=true"
semjoin run --split dev --matchers "cascade"
```
or string them together:
```sh
semjoin run --split dev --matchers "autofj;acronym:residual=true;cascade"
```

## Adding a matcher

Implement `match(task, ctx) -> set[(left, right)]` and register it in
`semjoin/matchers/__init__.py`:

```python
class MyMatcher:
    def __init__(self, batch_size: int = 20):
        self.batch_size = batch_size

    def match(self, task, ctx):
        with ctx.stage("profile"):
            ...
        with ctx.stage("llm_residual"):
            reply = ctx.llm.complete(prompt)   # calls and tokens are counted
        ctx.annotate(strategy="fuzzy")         # a router records its choice here
        return pairs
```

Always call the LLM through `ctx.llm` so calls and tokens are recorded.
Stage times are exclusive, so nested stages are not double-counted.
