"""Generate the vanilla50 group with the existing FramePack-F1 batch runner."""

from pathlib import Path
import sys


GROUP = "vanilla50"
# Shared generation and model settings are in experiments/f1_i2v_config.py.
CONFIG = dict(backend="huawei", mode="None", steps=13)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from run_f1_i2v import GROUPS, main

    GROUPS[GROUP] = CONFIG
    sys.argv[1:] = ["run", *sys.argv[1:], "--group", GROUP]
    sys.exit(main())
