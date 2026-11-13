"""Validate the paired manifest and capture the exact input/code/weight identities."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from f1_i2v_config import BACKENDS, GROUPS, PARAMS, REPOS, ROOT


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_samples(dataset, image_group):
    from PIL import Image

    manifest = dataset / "generation_manifest.draft.jsonl"
    metadata = dataset / "vbench2_i2v_full_info.json"
    manifest_bytes, metadata_bytes = manifest.read_bytes(), metadata.read_bytes()
    rows = [json.loads(line) for line in manifest_bytes.splitlines()]
    entries = json.loads(metadata_bytes)
    if len(entries) != 1118 or len(rows) != 5590:
        raise ValueError("Expected 1,118 metadata entries and 5,590 samples")
    image_root = (dataset / "image_suite" / image_group).resolve(strict=True)
    ids, names, seeds, images = set(), set(), set(), {}
    for row in rows:
        pair, sample, seed = (row[key] for key in ("pair_index", "sample_index", "seed"))
        if (type(pair) is not int or not 0 <= pair < 1118
                or type(sample) is not int or not 0 <= sample < 5
                or type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError(f"Invalid indices or seed: {row}")
        name, prompt = row["image_name"], row["prompt"]
        entry = entries[pair]
        generation_id = f"pair{pair:04d}_sample{sample}"
        eval_name = f"{prompt}-{sample}.mp4"
        if (row["generation_id"] != generation_id or generation_id in ids
                or eval_name in names or (pair, seed) in seeds
                or name != entry["image_name"] or prompt != entry["prompt_en"]
                or row["dimensions"] != entry["dimension"] or row["image_type"] != entry["image_type"]
                or not prompt.strip() or "/" in eval_name or "\\" in eval_name
                or "\0" in eval_name or len(os.fsencode(eval_name)) > 255):
            raise ValueError(f"Invalid or duplicate manifest record: {row}")
        ids.add(generation_id)
        names.add(eval_name)
        seeds.add((pair, seed))
        if name not in images:
            path = (image_root / name).resolve(strict=True)
            if path.parent != image_root or not path.is_file():
                raise ValueError(f"Invalid input image path: {path}")
            with Image.open(path) as image:
                image.load()
                width, height = image.size
            images[name] = dict(path=str(path), sha256=sha256(path), width=width, height=height)
        row.update(image=images[name], eval_name=eval_name)
    fingerprints = {path.name: hashlib.sha256(content).hexdigest()
                    for path, content in ((manifest, manifest_bytes), (metadata, metadata_bytes))}
    return sorted(rows, key=lambda row: row["generation_id"]), fingerprints


def snapshots(cache, verify_blobs=False):
    """Hugging Face LFS names are SHA-256 IDs. Optionally verify bytes against IDs."""
    identities, paths, verified = {}, {}, {}
    for repo, (revision, components) in REPOS.items():
        hub = cache / ("models--" + repo.replace("/", "--"))
        root = hub / "snapshots" / revision
        files = {}
        for component in components:
            folder = root / component
            if not folder.is_dir():
                raise ValueError(f"Missing required checkpoint component: {folder}")
            members = sorted(p for p in folder.rglob("*") if not p.is_dir())
            if not members:
                raise ValueError(f"Empty checkpoint component: {folder}")
            for path in members:
                target = path.resolve(strict=True)
                size = target.stat().st_size
                if not size:
                    raise ValueError(f"Empty checkpoint file: {path}")
                if path.name.endswith(".index.json"):
                    index = json.loads(path.read_bytes())
                    for shard in set(index["weight_map"].values()):
                        if not (path.parent / shard).is_file():
                            raise ValueError(f"Missing checkpoint shard: {path.parent / shard}")
                if path.suffix == ".safetensors":
                    with target.open("rb") as source:
                        header_size = int.from_bytes(source.read(8), "little")
                        if not 0 < header_size <= min(size - 8, 100_000_000):
                            raise ValueError(f"Invalid safetensors header: {path}")
                        header = json.loads(source.read(header_size))
                    end = max(item["data_offsets"][1] for key, item in header.items()
                              if key != "__metadata__")
                    if end + header_size + 8 != size:
                        raise ValueError(f"Checkpoint size differs from tensor data offsets: {path}")
                lfs_name = target.name
                is_lfs = target.parent == hub / "blobs" and len(lfs_name) == 64 and all(
                    char in "0123456789abcdef" for char in lfs_name)
                if not is_lfs or verify_blobs:
                    if target not in verified:
                        verified[target] = sha256(target)
                    if is_lfs and verified[target] != lfs_name:
                        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
                files[str(path.relative_to(root))] = dict(
                    size=size, sha256=lfs_name if is_lfs else verified[target],
                    identity_source="hub_lfs_blob" if is_lfs else "file_sha256")
        paths[repo] = root
        identities[repo] = dict(revision=revision, files=files)
    return paths, identities


def binary_version(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} is required on PATH")
    return subprocess.check_output([path, "-version"], text=True).splitlines()[0]


def provenance(args, rows, fingerprints, weights, device):
    source_files = {}
    for backend, root in BACKENDS.items():
        sources = [root / "demo_gradio_f1.py", *sorted((root / "diffusers_helper").rglob("*.py"))]
        if not (root / "diffusers_helper" / "device.py").is_file():
            raise ValueError(f"Missing device backend: {root}")
        source_files[backend] = dict(root=str(root), files={
            str(path.relative_to(root)): sha256(path) for path in sources})
    source_files["runner"] = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*f1_i2v*.py"))}
    source_files["pilot"] = sha256(ROOT / "tools" / "run_f1_i2v_pilot.py")
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    return dict(
        schema=1, dataset=str(args.dataset), image_group=args.image_group, input_hashes=fingerprints,
        image_count=len({row["image_name"] for row in rows}),
        code=source_files, weights=weights, hf_cache=str(args.hf_cache), hardware=device,
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                         packages=packages, ffmpeg=binary_version("ffmpeg"),
                         ffprobe=binary_version("ffprobe"),
                         variables={key: value for key, value in os.environ.items()
                                    if key.startswith(("ASCEND_", "HCCL_", "ACL_", "PYTORCH_NPU_"))}),
        params=PARAMS, groups=GROUPS, smoke_limit=args.smoke_limit,
    )
