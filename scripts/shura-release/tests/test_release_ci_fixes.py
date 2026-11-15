import re
import unittest
from pathlib import Path

REL = Path(__file__).resolve().parents[1]
ROOT = REL.parents[1]
WF = ROOT / ".github" / "workflows" / "shura-release.yml"
CI = REL / "ci-build.sh"
ENV = REL / "vulkan-sdk.env"
TOUCH = ROOT / "scripts" / "shura-touchpoints.py"


def read_env():
    d = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            k, v = line.split("=", 1)
            d[k] = v
    return d


def job_block(text, name):
    m = re.search(r"^  " + re.escape(name) + r":\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)", text, re.M | re.S)
    assert m, name
    return m.group(1)


class ReleaseCiFixes(unittest.TestCase):
    def test_vulkan_sdk_env(self):
        e = read_env()
        self.assertTrue(e["VULKAN_SDK_URL"].startswith("https://"))
        self.assertIn("/" + e["VULKAN_SDK_VERSION"] + "/", e["VULKAN_SDK_URL"])
        self.assertRegex(e["VULKAN_SDK_SHA256"], r"^[0-9a-f]{64}$")
        self.assertRegex(e["VULKAN_SDK_BYTES"], r"^[0-9]+$")

    def test_ci_build_packages(self):
        text = CI.read_text(encoding="utf-8")
        line = next(l for l in text.splitlines() if "pkgs+=" in l and "vulkan" in l)
        self.assertNotIn("glslc", line)
        self.assertIn("spirv-headers", line)
        self.assertIn("libvulkan-dev", line)
        common = next(l for l in text.splitlines() if l.startswith("common=("))
        self.assertIn("python3-tomli", common)

    def test_ci_build_pins_the_sdk(self):
        text = CI.read_text(encoding="utf-8")
        self.assertIn("vulkan-sdk.env", text)
        self.assertIn("VULKAN_SDK_SHA256", text)
        self.assertLess(text.index("vulkan-sdk.env"), text.index('"$here/build.sh"'))

    def test_container_jobs_use_bash(self):
        text = WF.read_text(encoding="utf-8")
        for job in ("build-hip", "build-cuda"):
            block = job_block(text, job)
            self.assertRegex(block, r"    defaults:\n      run:\n        shell: bash\n", job)
            self.assertLess(block.index("defaults:"), block.index("    steps:"), job)

    def test_touchpoints_has_tomli_fallback(self):
        text = TOUCH.read_text(encoding="utf-8")
        self.assertIn("tomllib", text)
        self.assertIn("tomli", text)


if __name__ == "__main__":
    unittest.main()
