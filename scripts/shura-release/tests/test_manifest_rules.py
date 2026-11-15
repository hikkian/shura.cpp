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


class HipRuntimeRules(unittest.TestCase):
    def hip(self, runtime, built_major=7):
        rel = copy.deepcopy(GOOD["releases"][0])
        rel.update({"id": "shura-1.0.0-hip-linux-x86_64", "family": "HIP", "gpu_targets": ["gfx1101"], "runtime": runtime,
                    "parts": ["vram_governor", "gqa_kernel", "mtp", "moe_cache"],
                    "built_with": {"commit": "abcdef1", "cmake_flags": ["-DGGML_NATIVE=OFF", "-DLLAMA_OPENSSL=OFF"],
                                   "runtime": {"rocm_major": built_major}}})
        rel["assets"][0]["name"] = "shura-cpp-1.0.0-hip-linux-x86_64.tar.gz"
        rel["assets"][0]["url"] = mf.RELEASE_BASE.format(version="1.0.0") + "/shura-cpp-1.0.0-hip-linux-x86_64.tar.gz"
        return {"schema": 1, "releases": [rel]}

    def test_a_range_with_both_ends_and_the_built_major_is_fine(self):
        self.assertEqual(probs(self.hip({"rocm_min": "7.0", "rocm_max": "7.99"}), "rocm"), [])

    def test_a_range_with_one_end_is_refused(self):
        self.assertTrue(probs(self.hip({"rocm_min": "7.0"}), "HIP needs both rocm_min and rocm_max"))

    def test_a_min_of_another_major_than_the_build_is_refused(self):
        self.assertTrue(probs(self.hip({"rocm_min": "6.2", "rocm_max": "7.99"}, built_major=7), "made with major 7"))


if __name__ == "__main__":
    unittest.main()
