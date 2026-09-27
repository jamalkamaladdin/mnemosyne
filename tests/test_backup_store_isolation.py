"""Backups must never overwrite each other or mix stores.

``create_backup`` used to name every file ``mnemosyne_backup_%Y%m%d_%H%M%S``
and write it with ``gzip.open(..., "wb")`` into one shared directory, so a
second backup in the same second, from the same or from another store,
silently replaced the first. A targeted ``mnemosyne reindex --db/--bank``
backup also landed next to the default store's backups, where
``list_backups``, ``rotate_backups`` and ``emergency_restore`` treated it as a
default-store snapshot.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from mnemosyne import cli
from mnemosyne.core import beam
from mnemosyne.dr import recovery

NEW_NAME = re.compile(r"mnemosyne_backup_\d{8}_\d{6}_\d{6}(_\d{2})?\.db\.gz")


def _isolate(monkeypatch, tmp_path):
    """Point HOME, the data dir and the backup dir into tmp_path."""
    data_dir = tmp_path / "data"
    backup_root = tmp_path / "backups"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_BACKUP_DIR", str(backup_root))
    return backup_root, data_dir / "mnemosyne.db"


def _make_store(path: Path, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS marker (label TEXT)")
    conn.execute("INSERT INTO marker VALUES (?)", (label,))
    conn.commit()
    conn.close()
    return path


def _freeze_clock(monkeypatch, *instants):
    """Make ``recovery.datetime.now()`` return ``instants`` in order, then the last."""
    queue = list(instants)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(recovery, "datetime", _Clock)


def _dump(path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read()


def test_same_second_backups_of_one_store_both_survive(monkeypatch, tmp_path):
    backup_root, default_db = _isolate(monkeypatch, tmp_path)
    _make_store(default_db, "first")
    _freeze_clock(monkeypatch, datetime(2026, 9, 27, 12, 0, 0, 123456))

    first = recovery.create_backup()
    first_bytes = Path(first["backup_path"]).read_bytes()
    _make_store(default_db, "second")
    second = recovery.create_backup()

    assert first["backup_path"] != second["backup_path"]
    assert Path(first["backup_path"]).read_bytes() == first_bytes
    assert "'second'" not in _dump(first["backup_path"])
    assert "'second'" in _dump(second["backup_path"])
    assert Path(first["metadata_path"]).is_file()
    assert Path(second["metadata_path"]).is_file()
    assert [b["file"] for b in recovery.list_backups()] == [
        second["backup_path"],
        first["backup_path"],
    ]


def test_targeted_reindex_backup_stays_out_of_default_backups(monkeypatch, tmp_path, capsys):
    backup_root, default_db = _isolate(monkeypatch, tmp_path)
    _make_store(default_db, "default")
    other_db = _make_store(tmp_path / "elsewhere" / "work.db", "other")
    _freeze_clock(
        monkeypatch,
        datetime(2026, 9, 27, 12, 0, 0, 0),
        datetime(2026, 9, 27, 12, 0, 1, 0),
    )
    default_backup = recovery.create_backup()
    monkeypatch.setattr(
        beam,
        "reindex_vectors",
        lambda conn, progress=None: {"model": "fake", "dim": 4},
    )

    cli.cmd_reindex(["--db", str(other_db), "--yes"])

    digest = hashlib.sha256(os.fsencode(str(other_db.resolve()))).hexdigest()[:8]
    store_dir = backup_root / "stores" / f"work-{digest}"
    targeted = sorted(store_dir.glob("mnemosyne_backup_*.db.gz"))
    assert len(targeted) == 1
    assert f"Backup created: {targeted[0]}" in capsys.readouterr().out
    meta = json.loads(targeted[0].with_suffix(".gz.json").read_text())
    assert meta["source_db"] == str(other_db.resolve())

    assert [b["file"] for b in recovery.list_backups()] == [default_backup["backup_path"]]
    assert recovery.health_check()["backups"]["total"] == 1

    restored = recovery.emergency_restore()
    assert restored["backup_used"] == default_backup["backup_path"]
    conn = sqlite3.connect(str(default_db))
    assert conn.execute("SELECT label FROM marker").fetchall() == [("default",)]
    conn.close()

    rotated = recovery.rotate_backups(keep=1)
    assert rotated["total_backups"] == 1
    assert rotated["deleted"] == 0
    assert targeted[0].is_file()
    assert Path(default_backup["backup_path"]).is_file()


def test_default_store_backup_location_and_listing_unchanged(monkeypatch, tmp_path):
    backup_root, default_db = _isolate(monkeypatch, tmp_path)
    _make_store(default_db, "default")
    backup_root.mkdir(parents=True)
    legacy = backup_root / "mnemosyne_backup_20260101_000000.db.gz"
    legacy.write_bytes(gzip.compress(b"-- legacy\n"))

    result = recovery.create_backup()

    backup_path = Path(result["backup_path"])
    assert backup_path.parent == backup_root
    assert NEW_NAME.fullmatch(backup_path.name)
    assert not (backup_root / "stores").exists()
    assert result["source_db"] == str(default_db.resolve())
    assert json.loads(Path(result["metadata_path"]).read_text())["source_db"] == str(
        default_db.resolve()
    )
    assert [b["name"] for b in recovery.list_backups()] == [backup_path.name, legacy.name]
