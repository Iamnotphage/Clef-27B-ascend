from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from memory_budget import MemoryBudget


GIB = 1 << 30


class MemoryBudgetTest(TestCase):
    def test_v1_shares_incremental_headroom_and_stops_on_pressure(self):
        with TemporaryDirectory() as temp:
            proc = Path(temp)
            mount = proc / "memory"
            mount.mkdir()
            (proc / "self").mkdir()
            (proc / "meminfo").write_text("MemTotal: 134217728 kB\nMemAvailable: 104857600 kB\n")
            (proc / "self/cgroup").write_text("4:memory:/\n")
            (proc / "self/mountinfo").write_text(f"1 0 0:1 / {mount} rw - cgroup memory rw,memory\n")
            (mount / "memory.limit_in_bytes").write_text(str(128 * GIB))
            (mount / "memory.stat").write_text(f"hierarchical_memory_limit {128 * GIB}\n")
            (mount / "memory.usage_in_bytes").write_text(str(77 * GIB))
            (mount / "memory.failcnt").write_text("0")
            budget = MemoryBudget(41 << 29, proc=proc, shared_reserve=10 * GIB, partners=2)

            self.assertEqual(budget(0), 41 << 29)
            (mount / "memory.usage_in_bytes").write_text(str(116 * GIB))
            self.assertEqual(budget(4 * GIB), 5 * GIB)
            (mount / "memory.failcnt").write_text("1")
            self.assertEqual(budget(4 * GIB), 0)

    def test_v2_parent_high_and_host_headroom(self):
        with TemporaryDirectory() as temp:
            proc = Path(temp)
            mount = proc / "cgroup"
            leaf = mount / "child"
            leaf.mkdir(parents=True)
            (proc / "self").mkdir()
            (proc / "meminfo").write_text("MemTotal: 134217728 kB\nMemAvailable: 104857600 kB\n")
            (proc / "self/cgroup").write_text("0::/child\n")
            (proc / "self/mountinfo").write_text(f"1 0 0:1 / {mount} rw - cgroup2 cgroup rw\n")
            for path in (mount, leaf):
                (path / "memory.max").write_text("max")
                (path / "memory.high").write_text("max")
                (path / "memory.current").write_text(str(70 * GIB))
                (path / "memory.events").write_text("high 0\nmax 0\noom 0\noom_kill 0\n")
            (mount / "memory.high").write_text(str(96 * GIB))
            budget = MemoryBudget(41 << 29, proc=proc, shared_reserve=10 * GIB, partners=2)

            self.assertEqual(budget(0), 8 * GIB)
            (proc / "meminfo").write_text("MemTotal: 134217728 kB\nMemAvailable: 6291456 kB\n")
            self.assertEqual(budget(0), GIB)
