"""Persist per-sample transaction intent before publishing either MP4."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

from f1_i2v_config import PARAMS
from f1_i2v_inputs import sha256
from f1_i2v_media import probe


def present(path):
    return path.exists() or path.is_symlink()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lock(path):
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Already locked by another runner: {path}") from error
        yield


class Journal:
    def __init__(self, run, group, params, run_id):
        self.run, self.group, self.params, self.run_id = run, group, params, run_id
        self.root = run / group
        self.records = self.root / "records"
        self.staging = self.root / ".staging"

    def setup(self):
        for directory in (self.records, self.staging, self.root / "raw", self.root / "eval"):
            directory.mkdir(parents=True, exist_ok=True)
            if directory.is_symlink() or directory.resolve() != directory:
                raise RuntimeError(f"Output directory must be local to this run: {directory}")

    def record_path(self, row):
        return self.records / f"{row['generation_id']}.json"

    def stage(self, row):
        return self.staging / row["generation_id"]

    def describe(self, row):
        return dict(run_id=self.run_id, group=self.group, sample=row, params=self.params,
                    paths={
                        "raw": f"{self.group}/raw/{row['generation_id']}.mp4",
                        "eval": f"{self.group}/eval/{row['eval_name']}",
                    })

    def get(self, row):
        path = self.record_path(row)
        if not present(path):
            return None
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Invalid journal record: {path}")
        record = json.loads(path.read_bytes())
        if any(record.get(key) != value for key, value in self.describe(row).items()):
            raise ValueError(f"Stale record identity: {path}")
        if record.get("status") not in {"pending", "complete", "failed", "oom", "stopped"}:
            raise ValueError(f"Unknown record status: {path}")
        return record

    def assert_unpublished(self, row):
        for relative in self.describe(row)["paths"].values():
            if present(self.run / relative):
                raise ValueError(f"Existing untracked output requires inspection: {self.run / relative}")

    def _check(self, record, kind, path, *, decode=True):
        artifact = record["artifacts"][kind]
        if path.is_symlink() or not path.is_file() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"Missing or changed output: {path}")
        if decode:
            return probe(path, PARAMS[f"{kind}_fps"], artifact["width"], artifact["height"])

    def verify(self, record):
        for kind, relative in record["paths"].items():
            self._check(record, kind, self.run / relative)

    def prepare(self, row, timing):
        stage = self.stage(row)
        record = dict(self.describe(row), status="pending", timing=timing, artifacts={})
        for kind in record["paths"]:
            path = stage / f"{kind}.mp4"
            data = probe(path, PARAMS[f"{kind}_fps"], timing["width"], timing["height"])
            with path.open("rb") as video:
                os.fsync(video.fileno())
            record["artifacts"][kind] = dict(data, sha256=sha256(path))
        fsync_dir(stage)
        fsync_dir(self.staging)
        atomic_json(self.record_path(row), record)
        return record

    def publish(self, row, record):
        stage = self.stage(row)
        for kind, relative in record["paths"].items():
            source, destination = stage / f"{kind}.mp4", self.run / relative
            self._check(record, kind, source, decode=False)
            if present(destination):
                if destination.is_symlink() or not destination.samefile(source):
                    raise ValueError(f"Existing video collides with prepared publication: {destination}")
            else:
                os.link(source, destination)
            fsync_dir(destination.parent)
        record = dict(record, status="complete")
        atomic_json(self.record_path(row), record)
        shutil.rmtree(stage)
        return record

    def record_failure(self, row, status, error):
        old = self.get(row) or {}
        attempts = old.get("attempts", [])
        attempts.append(dict(status=status, error=error))
        atomic_json(self.record_path(row), dict(self.describe(row), status=status, attempts=attempts))

    def clear_scratch(self, row):
        stage = self.stage(row)
        if present(stage):
            if stage.is_symlink() or not stage.is_dir():
                raise ValueError(f"Unexpected scratch path: {stage}")
            shutil.rmtree(stage)

    def unknown_paths(self, rows):
        known = {self.run / relative for row in rows for relative in self.describe(row)["paths"].values()}
        known.update(self.record_path(row) for row in rows)
        return [str(path) for directory in (self.root / "raw", self.root / "eval", self.records)
                if directory.is_dir() for path in sorted(directory.iterdir()) if path not in known]
