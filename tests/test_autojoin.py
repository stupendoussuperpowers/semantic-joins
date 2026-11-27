"""Auto-Join web tables, offline with a tiny fake archive."""

import io
import json
import tarfile

from semjoin.datasets import autojoin, load_tasks
from semjoin.datasets.pairdir import read_task
from semjoin.splits import write_manifest

FILES = {
    "rows.txt": "State:Name\nsource\n",
    "source.csv": "State,Visitors\nNew Mexico,1\nUtah,2\nOhio,3\n",
    "target.csv": "Name,Income\nNewMexico,10\nUtah,20\nIowa,30\n",
    # "New Mexico" differs from the target column only in whitespace: repaired.
    # "Atlantis" is in neither column: dropped.
    "ground truth.csv": "source-State,source-Visitors,target-Name,target-Income\n"
    "New Mexico,1,New Mexico,10\nUtah,2,Utah,20\nAtlantis,9,Atlantis,90\n",
}


def _archive():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for table in ("AJ-parks", "FF-phone-1"):
            for name, text in FILES.items():
                data = text.encode()
                info = tarfile.TarInfo(f"{autojoin.PREFIX}{table}/{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_read_table_repairs_whitespace_and_drops_the_rest():
    left, right, gold, repaired, dropped = autojoin.read_table(FILES)
    assert left == ["New Mexico", "Utah", "Ohio"] and right == ["NewMexico", "Utah", "Iowa"]
    assert gold == {("New Mexico", "NewMexico"), ("Utah", "Utah")}
    assert (repaired, dropped) == (1, 1)


def test_build_keeps_only_web_tables_and_holds_them_out(tmp_path):
    autojoin.fetch(tmp_path, getter=lambda url, headers=None: _archive())
    assert autojoin.build(tmp_path) == ["parks"]  # FF-* tables are skipped
    task = read_task(tmp_path / "autojoin" / "parks", "autojoin")
    assert "Ohio" in task.left and "Iowa" in task.right  # unmatched values kept on both sides
    meta = json.loads((tmp_path / "autojoin" / "parks" / "meta.json").read_text())
    assert meta["gold_repaired"] == 1 and meta["gold_dropped"] == 1

    assignment = write_manifest(tmp_path / "splits.json", load_tasks(["autojoin"], tmp_path), dev_fraction=0.9)
    assert assignment == {"autojoin/parks": "test"}
