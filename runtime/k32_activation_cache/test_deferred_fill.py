from contextlib import contextmanager
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
from weakref import ref

from activation_cache import ActivationCache
from deferred_fill import DeferredFill, MAX_ROWS, PINNED_BYTES, SNAPSHOT_BYTES, WIDTH
from step_protocol import run_candidate


class FakeTensor:
    def __init__(self, runtime, shape, device, *, pinned=False, storage=None):
        self.runtime, self.shape, self.device, self.pinned = runtime, shape, device, pinned
        self.dtype = runtime.bfloat16
        self.storage = self if storage is None else storage

    def untyped_storage(self):
        return self.storage

    def nbytes(self):
        return self.numel() * 2

    def data_ptr(self):
        return id(self)

    def numel(self):
        return self.shape[0] * (self.shape[1] if len(self.shape) == 2 else 1)

    def element_size(self):
        return 2

    def stride(self):
        return WIDTH, 1

    def is_pinned(self):
        return self.pinned

    def __getitem__(self, part):
        count = len(range(self.shape[0])[part])
        pinned = self.pinned and not (part.start and self.runtime.fail == "pinned_view")
        return FakeTensor(self.runtime, (count, *self.shape[1:]), self.device,
                          pinned=pinned, storage=self.storage)

    def view(self, shape):
        return FakeTensor(self.runtime, shape, self.device, pinned=self.pinned, storage=self.storage)

    def copy_(self, source, non_blocking=False):
        label = {("npu", "npu"): "snapshot", ("cpu", "npu"): "d2h", ("cpu", "cpu"): "pageable"}
        self.runtime.add(label[self.device.type, source.device.type])
        return self


class Runtime:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []
        self.bfloat16 = object()
        self.device = SimpleNamespace(type="npu")
        self.cpu = SimpleNamespace(type="cpu")
        self.producer, self.copy = self.stream_obj("producer"), self.stream_obj("copy")
        self.npu = SimpleNamespace(current_stream=lambda _: self.producer,
                                   Stream=lambda **_: self.copy, Event=self.event, stream=self.stream)

    def add(self, call):
        self.calls.append(call)
        if call == self.fail or (self.fail == "fatal" and call in ("done", "drain_copy")):
            raise RuntimeError(call)

    def stream_obj(self, name):
        return SimpleNamespace(device=self.device, name=name, wait_event=lambda _: self.add("wait_ready"),
                               synchronize=lambda: self.add("drain_" + name))

    def event(self):
        return SimpleNamespace(record=lambda s: self.add("ready" if s is self.producer else "done"),
                               synchronize=lambda: self.add("wait_done"))

    @contextmanager
    def stream(self, stream):
        self.add("copy_enter")
        try:
            yield
        finally:
            self.add("copy_exit")

    def empty(self, size, *, device, dtype, pin_memory):
        return FakeTensor(self, (size,), self.cpu, pinned=pin_memory)

    def empty_like(self, x, *, pin_memory=False):
        return FakeTensor(self, x.shape, x.device, pinned=pin_memory)

    def boundary(self, rows=3):
        return tuple(FakeTensor(self, (rows, WIDTH), self.device) for _ in range(2))


class DeferredFillTest(TestCase):
    def init(self, fail=None):
        runtime = Runtime(fail)
        budget = Mock(return_value=PINNED_BYTES + SNAPSHOT_BYTES + (4 << 20))
        fill = DeferredFill(runtime, lambda x: 2, budget)
        fill.cache = ActivationCache(fill.entry_limit)
        return runtime, budget, fill

    def test_after_norm_prepare_waits_then_commits_without_reading_budget(self):
        runtime, budget, fill = self.init()
        pair = runtime.boundary()
        layers = [lambda **_: pair for _ in range(64)]
        key = b"k" * 32
        votes = []

        def exchange(digest, ready):
            votes.append(ready)
            self.assertEqual(digest, key)
            if len(votes) == 2:
                self.assertEqual(runtime.calls[-4:], ["norm", "wait_done", "pageable", "pageable"])
                with fill.cache.lease(key) as value:
                    self.assertIsNone(value)
                budget.side_effect = AssertionError("commit must not read capacity")
            return True, ready

        def norm(*_):
            runtime.add("norm")
            return "result", None

        result = run_candidate(layers, (), lambda: pair, norm, key=key, cache=fill.cache,
                               restore=Mock(), capture=Mock(), exchange=exchange, stock=Mock(), deferred=fill)
        self.assertEqual(result, ("result", False))
        self.assertEqual(votes, [False, True])
        self.assertEqual(runtime.calls[:9], ["snapshot", "snapshot", "ready", "copy_enter", "wait_ready",
                                             "d2h", "d2h", "done", "copy_exit"])
        self.assertEqual(fill.cache.snapshot(), (1, 0, 3 * 2 * 2 * WIDTH + 1024))
        self.assertEqual(fill.cache.published, 1)
        self.assertIsNone(fill.flight)

    def test_failed_event_records_retain_and_drain_submitted_queues(self):
        for failure, expected in (("ready", ["drain_producer"]),
                                  ("done", ["drain_copy", "drain_producer"])):
            with self.subTest(failure=failure):
                runtime, _, fill = self.init(failure)
                self.assertIsNone(fill.begin(b"x", *runtime.boundary()))
                self.assertEqual(runtime.calls[-len(expected):], expected)
                self.assertNotIn("wait_done", runtime.calls)
                self.assertIsNone(fill.flight)
                self.assertEqual(fill.cache.snapshot(), (0, 0, 0))

    def test_failed_queue_drain_keeps_events_sources_and_charge(self):
        runtime, _, fill = self.init("fatal")
        pair = runtime.boundary()
        handle = ref(pair[0])
        with self.assertRaisesRegex(RuntimeError, "worker recovery required"):
            fill.begin(b"x", *pair)
        del pair
        self.assertIsNotNone(handle())
        self.assertIsNotNone(fill.flight.ready)
        self.assertIsNotNone(fill.flight.done)
        self.assertFalse(fill.flight.done_recorded)
        self.assertGreater(fill.cache.snapshot()[2], 0)
        with self.assertRaisesRegex(RuntimeError, "previous flight"):
            fill.begin(b"y", *runtime.boundary())

    def test_budget_and_shape_bypass_and_interior_view_must_pin(self):
        runtime, budget, fill = self.init()
        self.assertIsNone(fill.begin(b"large", *runtime.boundary(MAX_ROWS + 1)))
        budget.return_value = PINNED_BYTES + SNAPSHOT_BYTES
        self.assertIsNone(fill.begin(b"budget", *runtime.boundary()))
        self.assertIsNone(fill.pinned)
        runtime.fail = "pinned_view"
        budget.return_value += 4 << 20
        self.assertIsNone(fill.begin(b"view", *runtime.boundary()))
        self.assertFalse(fill.pinned_ready)
        self.assertNotIn("d2h", runtime.calls)
        self.assertEqual(fill.cache.snapshot(), (0, 0, 0))

    def test_invalidation_or_negative_vote_discards_unpublished_values(self):
        for invalidate in (False, True):
            runtime, _, fill = self.init()
            ticket = fill.begin(b"x", *runtime.boundary())
            if invalidate:
                fill.cache.invalidate()
            self.assertIs(fill.prepare(ticket), not invalidate)
            with fill.cache.lease(ticket.key) as value:
                self.assertIsNone(value)
            fill.discard(ticket)
            self.assertEqual(fill.cache.snapshot(), (0, 0, 0))
            self.assertFalse(fill.cache.reveal(ticket))

    def test_suffix_exception_retires_copy_and_cancels_reservation(self):
        runtime, _, fill = self.init()
        pair = runtime.boundary()
        capture = Mock(side_effect=RuntimeError("suffix error"))
        layers = [lambda **_: pair for _ in range(64)]
        layers[-1] = capture
        vote = Mock(return_value=(True, False))
        with self.assertRaisesRegex(RuntimeError, "suffix error"):
            run_candidate(layers, (), lambda: pair, Mock(), key=b"a" * 32, cache=fill.cache,
                          restore=Mock(), capture=Mock(), exchange=vote, stock=Mock(), deferred=fill)
        vote.assert_called_once()
        self.assertEqual(runtime.calls[-1], "wait_done")
        self.assertEqual(fill.cache.snapshot(), (0, 0, 0))
