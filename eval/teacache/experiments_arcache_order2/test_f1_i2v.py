"""No-model regression tests for media, journal recovery, and evaluator mapping."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from f1_i2v_config import DATASET, GROUPS, PARAMS
from f1_i2v_inputs import load_samples
from f1_i2v_journal import Journal
from f1_i2v_media import ffmpeg, frame_hashes, probe, retime


class FormalRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.raw = cls.root / "raw.mp4"
        cls.evaluation = cls.root / "eval.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x64:r=30", "-frames:v", 145,
               "-c:v", "libx264", "-crf", 16, cls.raw)
        retime(cls.raw, cls.evaluation)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_dataset_and_evaluator_names(self):
        rows, hashes = load_samples(DATASET, "origin")
        self.assertEqual(len(rows) * len(GROUPS), 39130)
        self.assertEqual(len(set(row["eval_name"] for row in rows)), 5590)
        self.assertEqual({row["eval_name"] for row in rows},
                         {f"{entry['prompt_en']}-{idx}.mp4"
                          for entry in json.loads((DATASET / "vbench2_i2v_full_info.json").read_bytes())
                          for idx in range(5)})
        self.assertEqual(len(hashes), 2)

    def test_retime_preserves_all_frames_and_order(self):
        raw_info = probe(self.raw, PARAMS["raw_fps"], 64, 64)
        eval_info = probe(self.evaluation, PARAMS["eval_fps"], 64, 64)
        self.assertEqual(raw_info["frames"], eval_info["frames"])
        self.assertAlmostEqual(eval_info["duration"], 145 / 24)
        self.assertEqual(frame_hashes(self.raw), frame_hashes(self.evaluation))
        with self.assertRaises(ValueError):
            probe(self.evaluation, PARAMS["raw_fps"], 64, 64)

    def journal(self, directory):
        journal = Journal(directory, "vanilla100", GROUPS["vanilla100"], "test")
        journal.setup()
        row = dict(generation_id="pair0000_sample0", eval_name="prompt-0.mp4", seed=0)
        return journal, row

    def test_recovers_crash_after_first_video_publish(self):
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            journal, row = self.journal(Path(folder))
            stage = journal.stage(row)
            stage.mkdir()
            (stage / "raw.mp4").write_bytes(self.raw.read_bytes())
            (stage / "eval.mp4").write_bytes(self.evaluation.read_bytes())
            record = journal.prepare(row, dict(width=64, height=64, seconds=1))
            import os
            link = os.link

            def fail_second(source, destination):
                if destination.parent.name == "eval":
                    raise RuntimeError("synthetic interruption")
                link(source, destination)

            with mock.patch("f1_i2v_journal.os.link", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    journal.publish(row, record)
            self.assertEqual(journal.get(row)["status"], "pending")
            journal.publish(row, journal.get(row))
            result = journal.get(row)
            journal.verify(result)
            (journal.run / result["paths"]["raw"]).unlink()
            with self.assertRaisesRegex(ValueError, "Missing or changed"):
                journal.verify(result)

    def test_collision_and_failed_sample(self):
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            journal, row = self.journal(Path(folder))
            path = journal.run / journal.describe(row)["paths"]["raw"]
            path.write_bytes(b"pre-existing")
            with self.assertRaisesRegex(ValueError, "untracked"):
                journal.assert_unpublished(row)
            self.assertEqual(path.read_bytes(), b"pre-existing")
            path.unlink()
            journal.record_failure(row, "oom", "mock NPU memory exhausted")
            self.assertEqual(journal.get(row)["status"], "oom")
            journal.assert_unpublished(row)


if __name__ == "__main__":
    unittest.main()
