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
semjoin download-autojoin # 31 real web-table tasks -> data/autojoin/ (held out: test only)
semjoin build-mapping     # ISO-based code/alias/typo/reorder/acronym and mixed-type tasks -> data/mapping/
semjoin split             # assign new tasks to dev/test in splits.json (commit it)
semjoin validate          # optional: check tasks and per-type split coverage
```

`data/wikidata/` (22 real code/alias/acronym/reorder tables) is committed,
because Wikidata changes over time; `download-wikidata` and `build-wikidata`
are only for updating it. Likewise `data/nyc/` (6 real NYC Open Data joins,
held out: test only) is committed; `download-nyc` re-queries it. Freyja tasks are labeled by hand in `data/freyja/`;
`download-freyja` fetches the FREYJA lake and ranks column pairs worth
labeling. See [docs/data-format.md](docs/data-format.md) for the format and
for each dataset.
`splits.json` is append-only: rerunning `split` only assigns tasks that are
not in it yet, stratified by dataset and mismatch type.

## Running

```sh
semjoin run --split dev --matchers "exact;normalized;fuzzy:threshold=0.7"
semjoin run --split test --datasets autofj --matchers "fuzzy" --llm ollama \
    --ollama-model qwen2.5:7b --ollama-url http://gpu-box:11434
semjoin summarize runs/<id>
semjoin summarize runs/<id> --router router --strategies "..."   # see Routing below
```

`--matchers` takes specs separated by `;`; arguments after `:` are passed to
the matcher's constructor. `--llm fake` (the default) uses an offline stub.
`--max-tasks-per-type N` and `--max-right N` run a small, fixed slice (at most
N tasks per dataset and mismatch type, N right values per task) for a quick
check before a full run.

Each run writes `runs/<timestamp>/`:

| file | contents |
|---|---|
| `config.json` | matchers, task list, LLM settings, git commit |
| `results.jsonl` | per (task, matcher): P/R/F1, hard-pair counts, LLM calls, tokens, runtime per stage, annotations |
| `errors.jsonl` | per (task, matcher): false positive and false negative pairs |
| `summary.md`, `summary.csv` | per (matcher, dataset): micro P/R/F1, macro F1, hard F1, LLM calls, tokens, runtime |
| `by_type.csv` | the same, per mismatch type |
| `routing.md` | with `--router`: routing accuracy and F1 regret vs. the best strategy per task |
| `routing.csv` | with `--router`: the same per-task rows plus the router's profile statistics |
| `cost_quality.png` | micro F1 vs. LLM calls per matcher |

**Hard F1** counts only pairs whose values differ after normalization, i.e.
the pairs a normalized equi-join cannot find.

## Matchers

Baselines: `exact`, `normalized`, `fuzzy`. Strategies: `autofj`, `transform`,
`acronym`, `llm_gen`, `llm`, `cascade`. Each strategy runs a normalized join, then
its own method, then (with `residual=true`) an LLM pass on what's left; see `semjoin/matchers/`.
`router` profiles each column pair and runs one of these strategies; see [Routing](#routing).

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

## Routing

`router` follows the proposal pipeline: a normalized join; a cheap profile of
the right values it left unmatched (no LLM; a sample of at most `sample=100`
of them against the whole left column); one strategy chosen by rules; then the
residual LLM pass. The first rule that passes wins:

| rule | test on the profile (default) | strategy |
|---|---|---|
| `acronym` | `acronym_rate >= acronym_min` (0.5) | `acronym` |
| `code` | shorter side: mean length `<= code_len` (4) and shape uniformity `>= code_uniformity` (0.8) | `llm_gen` |
| `reorder` | `token_set_equal >= reorder_min` (0.25) | `transform` |
| `typo` | `qgram_cos >= typo_cos` (0.5) | `autofj` |
| `fallback` | none of the above | `fallback` (`llm`) |

The statistics are defined in `semjoin/matchers/profiling.py`. Thresholds are
matcher arguments, e.g. `router:typo_cos=0.45,code_len=3`; tune them on the dev
split only. `router:residual=false,fallback=autofj` never calls the LLM.

To evaluate it (RQ1, RQ2), run it in the same run as `cascade` and every
strategy it can choose, under the exact specs it records:

```sh
semjoin run --split dev --matchers "router;cascade;acronym:residual=true;llm_gen:residual=true;transform:residual=true;autofj:residual=true;llm"
semjoin summarize runs/<id> --router router --strategies "acronym:residual=true;llm_gen:residual=true;transform:residual=true;autofj:residual=true;llm"
```

`routing.md` lists, per task, the rule that fired, the chosen strategy and the
oracle (the strategies tied for the best F1), with routing accuracy, mean F1
regret, and the macro F1 of the router and of perfect routing. `routing.csv`
adds the profile, for tuning thresholds. The router returns exactly the pairs
its chosen strategy returns on its own, so its regret is zero whenever its
choice ties the best. With `--llm fake` the LLM strategies find nothing, so
the oracle is only meaningful with a real LLM.

`semjoin tune` scores other thresholds on such a run without re-running
anything: it re-routes each task from the profile the router recorded and
looks up the picked strategy's result, so a whole grid takes seconds.

```sh
semjoin tune runs/<id>                     # built-in grid of all five thresholds
semjoin tune runs/<id> --grid "typo_cos=0.4,0.5;reorder_min=0.1,0.25" --objective accuracy
```

It also prints reference rows: the router as run and with the current
defaults, every fixed strategy, perfect routing (the best any
one-strategy-per-task router can do) and `cascade`. Tune on dev, then
evaluate the chosen thresholds on test once.

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
