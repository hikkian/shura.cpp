import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REL = Path(__file__).resolve().parents[1]
WF = REL.parents[1] / ".github" / "workflows" / "shura-release.yml"
SCRIPT = REL / "check-pins.sh"


def run(workflow_text, pins_dir=REL):
    with tempfile.TemporaryDirectory() as td:
        wf = Path(td) / "shura-release.yml"
        wf.write_text(workflow_text, encoding="utf-8")
        env = dict(os.environ, PINS_WORKFLOW=str(wf), PINS_DIR=str(pins_dir))
        return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)


class CheckPins(unittest.TestCase):
    def setUp(self):
        self.text = WF.read_text(encoding="utf-8")

    def test_the_real_workflow_passes(self):
        r = run(self.text)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_a_planted_different_cuda_digest_is_refused(self):
        bad = self.text.replace("a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97",
                                "b" * 64, 1)
        r = run(bad)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("differs from cuda-image.digest", r.stderr)

    def test_a_planted_different_rocm_digest_is_refused(self):
        bad = self.text.replace("42851dac319afce41cf993e25f95005b7f2cd0a0f6abd32ad8f25cd876ec56df", "c" * 64, 1)
        r = run(bad)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("differs from rocm-image.digest", r.stderr)

    def test_a_pin_me_placeholder_is_refused(self):
        bad = self.text.replace("42851dac319afce41cf993e25f95005b7f2cd0a0f6abd32ad8f25cd876ec56df", "PIN_ME", 1)
        r = run(bad)
        self.assertNotEqual(r.returncode, 0)

    def test_a_second_cuda_reference_is_refused(self):
        bad = self.text + "\n# nvidia/cuda:12.8.1-devel-ubuntu22.04\n"
        r = run(bad)
        self.assertNotEqual(r.returncode, 0)

    def test_a_tag_without_digest_is_refused(self):
        bad = self.text.replace("rocm/dev-ubuntu-22.04:7.2.1@sha256:", "rocm/dev-ubuntu-22.04:7.2.1#", 1)
        r = run(bad)
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
