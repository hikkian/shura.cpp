import hashlib
import json
import os
import tarfile
import tempfile
import unittest

import _fixture as fx

CUDA_NAMES = ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"]


class PackageRules(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(); self.addCleanup(self.td.cleanup)
        self.src, self.bin, self.out, self.bw = fx.make_tree(self.td.name)

    def test_cuda_extra_lib_not_in_attachment_a(self):
        libs = fx.fake_libs(self.td.name, CUDA_NAMES + ["libnccl.so.2"])
        with self.assertRaises(SystemExit) as c:
            fx.run_package("cuda", self.src, self.bin, self.out, self.bw, libs)
        self.assertIn("Attachment A", str(c.exception))

    def test_extra_lib_for_cpu_refused(self):
        libs = fx.fake_libs(self.td.name, ["libcudart.so.12"])
        with self.assertRaises(SystemExit) as c:
            fx.run_package("cpu", self.src, self.bin, self.out, self.bw, libs)
        self.assertIn("only for the cuda", str(c.exception))

    def test_foreign_lib_in_bin_dir_cpu(self):
        for n in ("libssl.so.3", "libamdhip64.so.7", "libvulkan.so.1"):
            p = os.path.join(self.bin, n); open(p, "w").write("x")
            with self.assertRaises(SystemExit) as c:
                fx.run_package("cpu", self.src, self.bin, self.out, self.bw)
            self.assertIn("foreign GPU or TLS library", str(c.exception))
            os.remove(p)

    def test_own_gpu_modules_allowed(self):
        open(os.path.join(self.bin, "libggml-vulkan.so"), "w").write("v")
        arc = fx.run_package("vulkan", self.src, self.bin, self.out, self.bw)
        self.assertTrue(os.path.isfile(arc))

    def test_cuda_needs_all_three_libs(self):
        libs = fx.fake_libs(self.td.name, ["libcudart.so.12", "libcublas.so.12"])
        with self.assertRaises(SystemExit) as c:
            fx.run_package("cuda", self.src, self.bin, self.out, self.bw, libs)
        self.assertIn("libcublasLt", str(c.exception))

    def test_cuda_alien_lib_in_bin_dir_refused(self):
        open(os.path.join(self.bin, "libnccl.so.2"), "w").write("x")
        libs = fx.fake_libs(self.td.name, CUDA_NAMES)
        with self.assertRaises(SystemExit) as c:
            fx.run_package("cuda", self.src, self.bin, self.out, self.bw, libs)
        self.assertIn("not our library", str(c.exception))

    def test_cuda_link_name_kept_in_archive(self):
        d = os.path.join(self.td.name, "linked"); os.makedirs(d)
        real = os.path.join(d, "libcudart.so.12.8.90"); open(real, "w").write("real cudart\n")
        names = ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"]
        links = []
        for n in names:
            target = real if n == "libcudart.so.12" else fx.fake_libs(self.td.name, [n])[0]
            ln = os.path.join(d, n)
            if target != ln:
                if os.path.lexists(ln): os.remove(ln)
                if n == "libcudart.so.12": os.symlink(real, ln)
                else: os.symlink(target, ln)
            links.append(ln)
        arc = fx.run_package("cuda", self.src, self.bin, self.out, self.bw, links)
        with tarfile.open(arc) as t:
            self.assertIn("shura-cpp-1.0.0-cuda/libcudart.so.12", t.getnames())
            self.assertNotIn("shura-cpp-1.0.0-cuda/libcudart.so.12.8.90", t.getnames())

    def test_cuda_eula_present_in_repo(self):
        self.assertTrue(os.path.isfile(os.path.join(fx.REL, "licenses-cuda", "NVIDIA-CUDA-EULA-12.8.txt")))

    def test_hip_rocm_major_must_match(self):
        json.dump({"commit": "abcdef1", "cmake_flags": ["-DGGML_NATIVE=OFF"], "runtime": {"rocm_major": 6}}, open(self.bw, "w"))
        with self.assertRaises(SystemExit):
            fx.run_package("hip", self.src, self.bin, self.out, self.bw)

    def test_cuda_archive_ok(self):
        libs = fx.fake_libs(self.td.name, CUDA_NAMES)
        arc = fx.run_package("cuda", self.src, self.bin, self.out, self.bw, libs)
        top = "shura-cpp-1.0.0-cuda"
        with tarfile.open(arc) as t:
            notice = t.extractfile(f"{top}/NOTICE").read().decode()
            self.assertTrue(notice.startswith(fx.NOTICE_TEXT + "\n"))
            self.assertIn("This product includes NVIDIA CUDA Toolkit 12.8 runtime libraries", notice)
            self.assertIn(f"{top}/licenses/NVIDIA-CUDA-EULA-12.8.txt", t.getnames())
            for p in libs:
                got = hashlib.sha256(t.extractfile(f"{top}/{os.path.basename(p)}").read()).hexdigest()
                self.assertEqual(got, hashlib.sha256(open(p, "rb").read()).hexdigest())
        self.assertEqual(open(os.path.join(self.src, "NOTICE")).read(), fx.NOTICE_TEXT)   # source NOTICE untouched


if __name__ == "__main__":
    unittest.main()
