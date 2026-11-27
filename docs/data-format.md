# Dataset format

Every dataset is a directory under `data/`, holding one subdirectory per task
(one column pair to join). All three datasets use the same layout, which is
the Auto-FuzzyJoin benchmark layout:

```
data/<dataset>/<task_id>/
    left.csv     id,title          one value per row
    right.csv    id,title          one value per row
    gt.csv       id_l,title_l,id_r,title_r
    meta.json    optional
```

- `title` holds the value. `id` only needs to be unique within its file;
  the harness matches on `title` strings, and duplicate titles are collapsed.
- `gt.csv` lists **every** true match between the two files. Matching is
  many-to-many: a value may appear in any number of rows, and a value may
  appear in none. Any pair not listed counts as a non-match, so a missing
  row counts against the matcher.
- Every `title_l` must appear in `left.csv` and every `title_r` in
  `right.csv`; loading fails otherwise.
- `meta.json` keys read by the harness:
  - `mismatch_type`: one of `code`, `alias`, `typo`, `reorder`, `acronym`,
    `fuzzy` (several small variations, as in Auto-FuzzyJoin) or `mixed`.
    Used for the per-type breakdown (RQ3) and to stratify splits; omit it
    only if unknown (`semjoin validate` warns).
  - `type_mix`: for `mixed` tasks, gold pair counts per type, e.g.
    `{"code": 40, "typo": 30, "alias": 30}`.
  - Any other keys (source, notes, labeler) are kept for reference only.

## Freyja labeling (`data/freyja/`)

`semjoin download-freyja` caches the FREYJA data lake (160 CSV tables, ~34 MB
zipped) and its column-level ground truth (which column pairs join), then
profiles every joinable pair and writes `data/freyja_candidates.csv`, most
promising first: string columns that are
not IDs or lists, where a normalized equi-join leaves 10-90% of values
unmatched, ranked by how many unmatched values have a near-miss on the other
side. Pick task pairs from the top of that file, and skim further down for
codes and aliases, which have no near-misses.

Freyja discovers joinable column pairs in a data lake. To turn a candidate
pair into a task:

1. Sample values from each column into `left.csv` and `right.csv`. Keep the
   sample small enough to label exhaustively (a few hundred values per side
   at most).
2. Label **all** matches between the two samples in `gt.csv`, not only the
   easy ones. Recall is only meaningful if the labels are exhaustive.
3. Name the task directory `<lake table>__<column>__<lake table>__<column>`
   (or any unique id) and add a `meta.json`, e.g.

   ```json
   {"mismatch_type": "alias", "source": "freyja", "labeler": "ae3029",
    "left_column": "orders.country", "right_column": "customers.cntry"}
   ```

4. Run `semjoin split` to give the new task a dev/test assignment.

## Type tags for untyped sources

Auto-FuzzyJoin has no mismatch types, so `semjoin download-autofj` tags every
gold pair with `semjoin/datasets/tagging.py` (reorder, acronym, fuzzy, typo or
alias) and gives the task the dominant tag if it covers at least 75% of pairs,
else `mixed`. The counts go to `type_mix`. Most autofj tasks come out `mixed`.
The same tagger can label Freyja tasks. It is a heuristic for the per-type
analysis, not ground truth.

## Raw cache (`data/raw/`)

Downloads go to `data/raw/<source>/` (gitignored) and builders read only from
there. Its `fetch.json` records each file's URL, sha256, size and fetch time,
plus what the source is pinned to (a commit, a query date). `semjoin validate`
re-hashes the cache. A new source adds `fetch` and `build` functions using
`semjoin.datasets.raw.RawCache` and a `download-<source>` command in
`semjoin/cli.py`.

## Auto-Join web tables (`data/autojoin/`): held out

31 pairs of real web tables from the Auto-Join benchmark (Zhu et al., PVLDB
2017), as redistributed by DTT, with value-level ground truth. They are the
real-world check: `semjoin split` puts all of them in `test`, so nothing is
tuned on them. Left and right are the declared join columns, unmatched values
included. Gold values that differ from their column only in whitespace are
repaired; others are dropped (`gold_repaired`, `gold_dropped` in
`meta.json`). Mismatch types come from the tagger, for analysis only.

## NYC Open Data (`data/nyc/`): held out

Six real-world joins whose gold comes from the data itself, not from hand
labels: agency names to acronyms and to former names (same row), 2012 vs.
2010 SAT school names (same DBN, sharing a distinctive word), neighborhood
names to NTA codes and abbreviations (same row), and NYCHA addresses to DOB
permit spellings of the same building (same BIN, same house number). The
docstring of `semjoin/datasets/nyc.py` lists each source and filter. Like
`autojoin` they are all in `test`. Tables are edited over time, so the built
tasks are committed and `meta.json` records dataset ids and the fetch date;
`semjoin download-nyc --refresh` re-queries them. Types are declared where
known by construction, tagged otherwise; `gold_dropped` counts pairs removed
by the filters. Derived gold is not perfect (a reused DBN can still share a
word), but it is exhaustive over the key.

## Wikidata (`data/wikidata/`)

One task per SPARQL table in `semjoin/datasets/queries/wikidata/`, registered
with its mismatch type, group and right-value pattern in `TABLES` in
`semjoin/datasets/wikidata.py`, which also documents the cleaning and
sampling. Each `meta.json` records the endpoint, fetch time and the sha256 of
the raw response and of the query. To add a table: write the `.rq` file
(columns `?item ?left ?right`), add it to `TABLES`, then
`semjoin download-wikidata --names <table>` and `semjoin build-wikidata --names <table>`.

## Mixed-type tasks (`data/mapping/mixed_*`)

Each task mixes mismatch types within one domain (e.g. countries as ISO
codes, official names, typos and "Emirates, United Arab"), with target
weights per type in `SPECS` in `semjoin/datasets/mixed.py`. Entities are
assigned to types by quota, so the realized `type_mix` matches the weights
to within a few pairs. Built offline from pycountry and the committed
`data/wikidata/` tasks by `semjoin build-mapping`.

Unlike `data/autofj/` and `data/mapping/`, which can be regenerated,
`data/freyja/` is hand-made and is committed to git.
