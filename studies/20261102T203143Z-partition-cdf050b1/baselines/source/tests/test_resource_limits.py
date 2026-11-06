import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from limited_run import run_limited


@unittest.skipIf(os.environ.get("CLL_RESOURCE_LIMITS"), "Launcher checks run separately to avoid nested job locks")
class ResourceLimitTests(unittest.TestCase):
    def test_child_limits_and_cpu_throttle_are_applied(self):
        code = """
import json, os, resource, time
start = time.process_time()
while time.process_time() - start < 0.6:
    sum(range(1000))
print(json.dumps({
    'affinity': list(os.sched_getaffinity(0)),
    'nice': os.getpriority(os.PRIO_PROCESS, 0),
    'address_limit': resource.getrlimit(resource.RLIMIT_AS)[0],
    'threads': os.environ['OPENBLAS_NUM_THREADS'],
}))
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code_return = run_limited([sys.executable, "-c", code], root, cpu_percent=25,
                                      address_mib=256, resident_mib=128, reserve_mib=0)
            self.assertEqual(code_return, 0)
            output = next(root.iterdir())
            child = json.loads((output / "output.log").read_text())
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(len(child["affinity"]), 1)
            self.assertGreaterEqual(child["nice"], 15)
            self.assertEqual(child["address_limit"], 256 * 2**20)
            self.assertEqual(child["threads"], "1")
            self.assertGreater(result["wall_seconds"], 1.8)

    def test_allocation_cannot_exceed_hard_address_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_limited([sys.executable, "-c", "bytearray(300 * 2**20)"], root,
                                 address_mib=128, resident_mib=96, reserve_mib=0)
            self.assertNotEqual(result, 0)
            output = next(root.iterdir())
            self.assertIn("MemoryError", (output / "output.log").read_text())

    def test_wall_limit_stops_child_and_retains_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_limited([sys.executable, "-c", "import time; time.sleep(30)"], root,
                                 address_mib=128, resident_mib=96, reserve_mib=0, wall_seconds=0.3)
            self.assertNotEqual(result, 0)
            output = next(root.iterdir())
            receipt = json.loads((output / "result.json").read_text())
            self.assertEqual(receipt["stop_reason"], "wall_time_limit")


if __name__ == "__main__":
    unittest.main()
