"""Generate, resume and audit seven paired FramePack-F1 VBench-I2V groups."""
import argparse
import ast
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from types import ModuleType

from f1_i2v_config import BACKENDS, DATASET, GROUPS, HF_CACHE, IMAGE_GROUPS, OUTPUTS, PARAMS
from f1_i2v_inputs import load_samples, provenance, sha256, snapshots
from f1_i2v_journal import Journal, atomic_json, lock, present
from f1_i2v_media import retime


def cli():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Use preflight before run. Run each group in its own process, with vanilla100 first. "
        "Reuse --run-id, --image-group, --device, and --smoke-limit to resume. "
        "Status reads recorded progress. Audit verifies every completed MP4. "
        "Retry failed samples with --retry-failed after fixing the cause."))
    parser.add_argument("action", choices=("preflight", "run", "status", "audit"))
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--run-id", help="New unique subdirectory name under --outputs")
    parser.add_argument("--image-group", choices=IMAGE_GROUPS)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    parser.add_argument("--hf-cache", type=Path, default=HF_CACHE)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--smoke-limit", type=int, choices=range(1, 9), metavar="1..8",
                        help="Test only the first N identical sample IDs in every group")
    parser.add_argument("--verify-weight-bytes", action="store_true",
                        help="Verify actual checkpoint bytes against pinned cache IDs during preflight")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue after ordinary per-sample failures. OOM always stops the group.")
    args = parser.parse_args()
    if args.action == "run" and (not args.group or not args.image_group):
        parser.error("run requires explicit --group and --image-group")
    if args.action == "preflight" and not args.image_group:
        parser.error("preflight requires explicit --image-group")
    if args.action != "preflight" and (not args.run_id or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", args.run_id)):
        parser.error("supply a unique --run-id using 1–96 letters, digits, dots, underscores or dashes")
    if args.smoke_limit and args.run_id and not args.run_id.startswith("f1-i2v-smoke-"):
        parser.error("smoke runs require an isolated --run-id starting with f1-i2v-smoke-")
    for name in ("dataset", "outputs", "hf_cache"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    return args


# Device, model loading and one-sample generation

class StopGeneration(RuntimeError):
    """Accelerator state requires operator attention before continuing."""


def is_oom(error):
    text = str(error).lower()
    return any(marker in text for marker in ("out of memory", "memory allocation failed",
                                              "acl_error_rt_memory_allocation", "507011"))


def accelerator(device):
    import torch
    import torch_npu  # Registers the NPU torch backend.

    if not torch.npu.is_available():
        raise StopGeneration("Ascend NPU unavailable")
    npu = torch.device(device)
    if npu.type != "npu" or npu.index is None:
        raise ValueError("Use an explicit NPU index, e.g. npu:0")
    torch.npu.set_device(npu)
    return torch, torch.npu, npu


def device_info(device):
    torch, api, npu = accelerator(device)
    _, total = api.mem_get_info(npu)
    return dict(device=device, name=api.get_device_name(npu), total_bytes=total,
                torch=torch.__version__, cann=getattr(torch.version, "cann", None),
                npu_driver=subprocess.check_output(["npu-smi", "info", "-t", "board", "-i",
                                                   str(physical_device())], text=True).strip())


def physical_device():
    """Get the physical card shown in npu-smi for the single-card server."""
    import re

    info = subprocess.check_output(["npu-smi", "info", "-l"], text=True)
    ids = re.findall(r"NPU ID\s*:\s*(\d+)", info)
    if len(ids) != 1:
        raise StopGeneration("Expected one physical NPU visible in npu-smi info")
    return int(ids[0])


def _assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets)


def _headless_source(source, snapshots, scratch):
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    cut = next(index for index, node in enumerate(module.body) if _assigns(node, "quick_prompts"))
    module.body = module.body[:cut]
    worker, = (node for node in module.body if isinstance(node, ast.FunctionDef)
               and node.name == "worker")
    handlers = [handler for statement in worker.body if isinstance(statement, ast.Try)
                for handler in statement.handlers if handler.type is None]
    if len(handlers) != 1:
        raise ValueError(f"Unexpected worker exception structure: {source}")
    handlers[0].body = [ast.Raise()]
    loaded_repos = set()
    for node in ast.walk(module):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pretrained" and node.args):
            continue
        candidate = node.args[0]
        if (isinstance(candidate, ast.Call) and ast.unparse(candidate.func) == "os.environ.get"
                and len(candidate.args) == 2 and isinstance(candidate.args[0], ast.Constant)
                and candidate.args[0].value == "FRAMEPACK_F1_MODEL"):
            candidate = candidate.args[1]
        if not isinstance(candidate, ast.Constant) or candidate.value not in snapshots:
            raise ValueError(f"Unexpected pretrained source in {source}: {ast.unparse(node.args[0])}")
        repo = candidate.value
        loaded_repos.add(repo)
        node.args[0] = ast.Constant(str(snapshots[repo]))
    if loaded_repos != set(snapshots):
        raise ValueError(f"F1 worker model IDs changed in {source}")
    folder, = (node for node in module.body if _assigns(node, "outputs_folder"))
    folder.value = ast.Constant(str(scratch))
    return compile(ast.fix_missing_locations(module), str(source), "exec")


class Queue:
    def top(self):
        return None

    def push(self, item):
        pass


@dataclass
class WorkerModels:
    group: dict
    torch: ModuleType
    api: ModuleType
    device: object
    ns: dict
    save: Callable


def load_models(group, snapshots, device, scratch):
    """Load the selected demo's existing models, before its Gradio UI."""
    torch, api, npu = accelerator(device)
    free, total = api.mem_get_info(npu)
    if free <= 60 * 1024**3:
        raise StopGeneration(f"High-VRAM guard: free {free / 1024**3:.2f} GiB "
                             f"of {total / 1024**3:.2f} GiB, require >60 GiB")
    os.environ.update(FRAMEPACK_HIGH_VRAM="1", FRAMEPACK_DEVICE=device, HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", GRADIO_ANALYTICS_ENABLED="False")
    root = BACKENDS[group["backend"]]
    source = root / "demo_gradio_f1.py"
    sys.path.insert(0, str(root))
    os.chdir(root)
    argv = [str(source), "--device", device]
    if group["backend"] == "tea":
        argv += ["--acceleration", "teacache", "--teacache-threshold", str(group["threshold"])]
    ns = {"__file__": str(source), "__name__": "__formal_f1_worker__"}
    previous, sys.argv = sys.argv, argv
    try:
        exec(_headless_source(source, snapshots, scratch), ns)
    finally:
        sys.argv = previous
    if ns.get("high_vram") is not True:
        raise StopGeneration("F1 selected low-VRAM mode")
    if group["backend"] == "taylor":
        original_init = ns["cache_init"]

        def checked_init(*args, **kwargs):
            cache, current = original_init(*args, **kwargs)
            for name in ("max_order", "fresh_threshold", "first_enhance"):
                if cache[name] != group[name]:
                    raise StopGeneration(f"Unexpected TaylorSeer {name}: {cache[name]}")
            return cache, current

        ns["cache_init"] = checked_init
    if group["mode"] == "ARCache":
        if ns.get("ARCACHE_ORDER") != group["order"]:
            raise StopGeneration("ARCache order differs from the frozen preset")
        model = ns["transformer"]
        initialize = model.initialize_arcache

        def initialize_arcache(**kwargs):
            if not kwargs["enable_arcache"] or kwargs["erc_order"] != group["order"]:
                raise StopGeneration("ARCache/ERC was not enabled as requested")
            return initialize(**kwargs, rel_l1_thresh=group["threshold"], ksi=group["ksi"])

        model.initialize_arcache = initialize_arcache
    sampler = ns["sample_hunyuan"]

    def sample_without_preview(**kwargs):
        kwargs["callback"] = None
        return sampler(**kwargs)

    ns["sample_hunyuan"] = sample_without_preview
    queue = Queue()
    ns["stream"] = type("Stream", (), {"input_queue": queue, "output_queue": queue})()
    return WorkerModels(group, torch, api, npu, ns, ns["save_bcthw_as_mp4"])


def worker(models, row, scratch, raw):
    """Run the original F1 worker for one manifest row and save the final MP4."""
    import numpy as np
    from PIL import Image

    group, params, ns = models.group, PARAMS, models.ns
    image_path = Path(row["image"]["path"])
    if sha256(image_path) != row["image"]["sha256"]:
        raise StopGeneration(f"Input changed after validation: {image_path}")
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB"))
    height, width = ns["find_nearest_bucket"](*pixels.shape[:2],
                                              resolution=params["resolution_bucket"])
    expected_shapes = [(37 + 36 * index, height, width) for index in range(params["sections"])]
    shapes = []

    def save_final(pixels, path, *, fps, crf):
        shape = tuple(pixels.shape[-3:])
        shapes.append(shape)
        if shape != expected_shapes[len(shapes) - 1] or fps != params["raw_fps"] or crf != params["crf"]:
            raise RuntimeError(f"Unexpected F1 video parameters: {shape}, fps={fps}, crf={crf}")
        if len(shapes) == len(expected_shapes):
            models.save(pixels, str(raw), fps=fps, crf=crf)

    ns.update(outputs_folder=str(scratch), save_bcthw_as_mp4=save_final)
    seed = row["seed"]
    random.seed(seed)
    np.random.seed(seed % 2**32)
    models.torch.manual_seed(seed)
    models.api.manual_seed_all(seed)
    models.api.synchronize(models.device)
    started = time.perf_counter()
    ns["worker"](pixels, row["prompt"], params["negative_prompt"], seed,
                 params["length_argument"], params["latent_window"], group["steps"], params["cfg"],
                 params["guidance"], params["rescale"], params["memory_preservation_gb"],
                 group["mode"], params["crf"])
    models.api.synchronize(models.device)
    elapsed = time.perf_counter() - started
    if shapes != expected_shapes or not raw.is_file():
        raise RuntimeError(f"Incomplete worker output: {shapes}")
    return dict(seconds=elapsed, width=width, height=height)


# Run setup, batch processing and command-line entry

def read_json(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular JSON file: {path}")
    return json.loads(path.read_bytes())


def fingerprint(value):
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def initialize_run(run, data, rows):
    # New runs and concurrent group processes share one immutable manifest.
    created = not present(run)
    if created:
        try:
            run.mkdir()
        except FileExistsError:
            created = False
    if run.is_symlink() or run.resolve() != run or not run.is_dir():
        raise ValueError(f"Unsafe run directory: {run}")
    marker = run / "provenance.json"
    manifest = run / "samples.json"
    with lock(run / ".run.lock"):
        if not present(marker):
            if not created:
                raise ValueError(f"Existing output directory lacks run provenance: {run}")
            atomic_json(manifest, rows)
            atomic_json(marker, data)
        if read_json(marker) != data or fingerprint(read_json(manifest)) != data["samples_sha256"]:
            raise ValueError("Existing run provenance differs from this invocation")


def report(run, group, groups, rows, run_key, *, audit):
    results = {}
    for name in [group] if group else groups:
        journal = Journal(run, name, groups[name], run_key)
        counts = dict(complete=0, pending=0, missing=0, failed=0, oom=0, stopped=0, stale=0)
        problems = []
        for row in rows:
            record = None
            try:
                record = journal.get(row)
                if record is None or record["status"] in {"failed", "oom", "stopped"}:
                    journal.assert_unpublished(row)
                elif audit and record["status"] == "complete":
                    journal.verify(record)
                status = record["status"] if record else "missing"
                counts[status] += 1
                if record and status in {"failed", "oom", "stopped"}:
                    problems.append(dict(id=row["generation_id"], attempts=record["attempts"]))
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                counts["stale"] += 1
                problems.append(dict(id=row["generation_id"], error=str(error)))
        unknown = journal.unknown_paths(rows)
        results[name] = dict(expected=len(rows), **counts, failed_or_stale=problems,
                             untracked_paths=unknown)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return all(summary["complete"] == summary["expected"] and not summary["untracked_paths"]
               for summary in results.values())


def run_group(args, run, rows, run_key, model_paths):
    journal = Journal(run, args.group, GROUPS[args.group], run_key)
    journal.setup()
    with lock(journal.root / ".group.lock"):
        unknown = journal.unknown_paths(rows)
        if unknown:
            raise ValueError(f"Untracked output or records in {journal.root}: {unknown}")
        work = []
        for row in rows:
            record = journal.get(row)
            if record and record["status"] == "pending":
                record = journal.publish(row, record)
            if record and record["status"] == "complete":
                journal.verify(record)
                journal.clear_scratch(row)
                continue
            journal.assert_unpublished(row)
            if not record or args.retry_failed:
                work.append(row)
        if not work:
            return report(run, args.group, GROUPS, rows, run_key, audit=False)
        warmup = journal.staging / ".warmup"
        if present(warmup):
            if warmup.is_symlink() or not warmup.is_dir():
                raise ValueError(f"Unsafe warm-up directory: {warmup}")
            shutil.rmtree(warmup)
        warmup.mkdir()
        try:
            try:
                models = load_models(GROUPS[args.group], model_paths, args.device, warmup)
                if not args.smoke_limit:
                    worker(models, rows[0], warmup, warmup / "discard.mp4")
            except Exception as error:
                journal.record_failure(work[0], "oom" if is_oom(error) else "stopped",
                                       f"Model initialization/warm-up: {error}")
                raise
        finally:
            shutil.rmtree(warmup)
        for row in work:
            stage = journal.stage(row)
            journal.clear_scratch(row)
            stage.mkdir()
            try:
                timing = worker(models, row, stage, stage / "raw.mp4")
                retime(stage / "raw.mp4", stage / "eval.mp4")
                record = journal.prepare(row, timing)
            except Exception as error:
                # prepare may have written intent before fsync or another IO error.
                old = journal.get(row)
                if old and old["status"] == "pending":
                    raise
                status = "oom" if is_oom(error) else "stopped" if isinstance(
                    error, StopGeneration) else "failed"
                if isinstance(error, subprocess.CalledProcessError):
                    detail = error.stderr or str(error)
                else:
                    detail = traceback.format_exc()
                journal.record_failure(row, status, detail)
                print(f"{status.upper()} {args.group}/{row['generation_id']}: {error}",
                      file=sys.stderr, flush=True)
                if status != "failed" or not args.continue_on_error:
                    raise
                continue
            journal.publish(row, record)
            print(f"COMPLETE {args.group}/{row['generation_id']} {timing['seconds']:.2f}s",
                  flush=True)
        return report(run, args.group, GROUPS, rows, run_key, audit=False)


def main():
    args = cli()
    if args.action in {"status", "audit"}:
        run = args.outputs / args.run_id
        data = read_json(run / "provenance.json")
        rows = read_json(run / "samples.json")
        if fingerprint(rows) != data["samples_sha256"]:
            raise ValueError("Run sample manifest hash mismatch")
        complete = report(run, args.group, data["groups"], rows, fingerprint(data),
                          audit=args.action == "audit")
        return 0 if complete or args.action == "status" else 1

    rows, hashes = load_samples(args.dataset, args.image_group)
    model_paths, weights = snapshots(args.hf_cache, args.verify_weight_bytes)
    for root in BACKENDS.values():
        _headless_source(root / "demo_gradio_f1.py", model_paths, args.outputs / ".unused-preflight")
    hardware = device_info(args.device)
    data = provenance(args, rows, hashes, weights, hardware)
    if args.smoke_limit:
        rows = rows[:args.smoke_limit]
    data["samples_sha256"] = fingerprint(rows)
    run_key = fingerprint(data)
    if args.action == "preflight":
        usage = shutil.disk_usage(args.outputs)
        print(json.dumps(dict(samples_per_group=len(rows), groups=GROUPS,
                              total=len(rows) * len(GROUPS), image_group=args.image_group,
                              hardware=hardware, output_root=str(args.outputs),
                              disk_free_bytes=usage.free, fingerprints=hashes,
                              weights={repo: item["revision"] for repo, item in weights.items()},
                              verified_weight_bytes=args.verify_weight_bytes, run_key=run_key),
                         ensure_ascii=False, indent=2))
        return 0
    run = args.outputs / args.run_id
    initialize_run(run, data, rows)
    return 0 if run_group(args, run, rows, run_key, model_paths) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Resume with the same run ID.", file=sys.stderr)
        sys.exit(130)
