from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import run_logging


class RunLoggingTests(unittest.TestCase):
    def test_metrics_and_latest_files_are_created(self) -> None:
        workspace_tmp = Path(__file__).resolve().parents[1] / "test_artifacts" / uuid.uuid4().hex
        workspace_tmp.mkdir(parents=True, exist_ok=True)

        original_logs_dir = run_logging.LOGS_DIR
        run_logging.LOGS_DIR = workspace_tmp
        try:
            paths = run_logging.create_run_log_paths("mnist")
            run_logging.init_metrics_log(paths.metrics_csv, ["epoch", "loss"])
            run_logging.append_csv_row(paths.metrics_csv, ["epoch", "loss"], {"epoch": 1, "loss": "0.1"})
            run_logging.write_latest_pointer(paths.latest_csv, paths.run_id, paths.metrics_csv)

            self.assertTrue(paths.metrics_csv.exists())
            self.assertTrue(paths.latest_csv.exists())
        finally:
            run_logging.LOGS_DIR = original_logs_dir
            shutil.rmtree(workspace_tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
