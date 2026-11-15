import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture as fx  # noqa: F401  (puts scripts/shura-release on sys.path)
import manifest as mf

NAME = "shura-cpp-1.0.0-cpu-linux-x86_64.tar.gz"
GOOD = {
    "schema": 1,
    "releases": [{
        "id": "shura-1.0.0-cpu-linux-x86_64", "engine": "shura.cpp", "version": "1.0.0", "family": "CPU", "arch": "x86_64",
        "gpu_targets": [], "runtime": {}, "glibc_min": "2.35", "parts": ["mtp", "moe_cache"],
        "built_with": {"commit": "abcdef1", "cmake_flags": ["-DGGML_NATIVE=OFF", "-DLLAMA_OPENSSL=OFF"]},
        "assets": [{"name": NAME, "url": mf.RELEASE_BASE.format(version="1.0.0") + "/" + NAME, "sha256": "0" * 64, "size": 50 * 1024 * 1024}],
    }],
}


def probs(m, word):
    return [p for p in mf.check(m) if word in p]


class ManifestRules(unittest.TestCase):
    def mk(self):
        return copy.deepcopy(GOOD)

    def test_baseline_clean(self):
        self.assertEqual(mf.check(self.mk()), [])

    def test_glibc_above_ceiling(self):
        m = self.mk(); m["releases"][0]["glibc_min"] = "2.43"
        self.assertTrue(probs(m, "above the ceiling 2.35"))

    def test_glibc_at_ceiling(self):
        self.assertEqual(probs(self.mk(), "glibc"), [])

    def test_url_placeholder(self):
        m = self.mk(); m["releases"][0]["assets"][0]["url"] = "https://example.invalid/" + NAME
        self.assertTrue(probs(m, "url must be exactly"))

    def test_url_template_ok(self):
        self.assertEqual(probs(self.mk(), "url"), [])

    def test_missing_openssl_off(self):
        m = self.mk(); m["releases"][0]["built_with"]["cmake_flags"] = ["-DGGML_NATIVE=OFF"]
        self.assertTrue(probs(m, "LLAMA_OPENSSL=OFF"))


if __name__ == "__main__":
    unittest.main()
