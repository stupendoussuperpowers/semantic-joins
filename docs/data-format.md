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
    or omit if mixed or unknown. Used for the per-type breakdown (RQ3).
  - Any other keys (source, notes, labeler) are kept for reference only.

## Freyja labeling (`data/freyja/`)

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

Unlike `data/autofj/` and `data/mapping/`, which can be regenerated,
`data/freyja/` is hand-made and is committed to git.
