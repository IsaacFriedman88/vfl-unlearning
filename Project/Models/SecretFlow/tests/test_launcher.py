from __future__ import annotations

import unittest

import run_vfl


class LauncherTests(unittest.TestCase):
    def test_expected_datasets_are_registered(self) -> None:
        self.assertEqual(set(run_vfl.RUNNERS), {"celeba", "mnist", "nih"})


if __name__ == "__main__":
    unittest.main()
