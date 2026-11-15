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
        self.assertIn("libvulkan1", line)
        self.assertNotIn("libvulkan-dev", line)
        common = next(l for l in text.splitlines() if l.startswith("common=("))
        self.assertIn("python3-tomli", common)

    def test_ci_build_pins_the_sdk(self):
        text = CI.read_text(encoding="utf-8")
        self.assertIn("vulkan-sdk.env", text)
        self.assertIn("VULKAN_SDK_SHA256", text)
        self.assertLess(text.index("vulkan-sdk.env"), text.index('"$here/build.sh"'))

    def test_ci_build_unpacks_headers_and_loader(self):
        text = CI.read_text(encoding="utf-8")
        self.assertIn("/x86_64/include/*", text)
        self.assertIn("lib/VulkanLoader/lib/libvulkan.so*", text)
        self.assertIn('VULKAN_SDK="$SRC/vulkan-sdk"; export VULKAN_SDK', text)

    def test_container_jobs_safe_directory(self):
        text = WF.read_text(encoding="utf-8")
        want = 'git config --global --add safe.directory "$GITHUB_WORKSPACE"'
        for job in ("build-hip", "build-cuda"):
            block = job_block(text, job)
            self.assertIn(want, block, job)
            self.assertLess(block.index(want), block.index("git init"), job)

    def test_container_jobs_use_bash(self):
        text = WF.read_text(encoding="utf-8")
        for job in ("build-hip", "build-cuda"):
            block = job_block(text, job)
            self.assertRegex(block, r"    defaults:\n      run:\n        shell: bash\n", job)
            self.assertLess(block.index("defaults:"), block.index("    steps:"), job)

    def test_ci_build_check_work_on_disk(self):
        text = CI.read_text(encoding="utf-8")
        self.assertIn('WORK="$SRC/check-work"', text)
        self.assertLess(text.index('WORK="$SRC/check-work"'), text.index('"$here/check.sh"'))

    def test_ci_build_removes_build_dir_after_package(self):
        text = CI.read_text(encoding="utf-8")
        self.assertIn('rm -rf "$BUILD_DIR"', text)
        i = text.index('rm -rf "$BUILD_DIR"')
        self.assertGreater(i, text.index("package.py"))
        self.assertLess(i, text.index('"$here/check.sh"'))

    def test_ci_build_cleans_apt(self):
        text = CI.read_text(encoding="utf-8")
        self.assertIn("apt-get clean", text)
        self.assertIn("rm -rf /var/lib/apt/lists/*", text)

    def test_touchpoints_has_tomli_fallback(self):
        text = TOUCH.read_text(encoding="utf-8")
        self.assertIn("tomllib", text)
        self.assertIn("tomli", text)


if __name__ == "__main__":
    unittest.main()
