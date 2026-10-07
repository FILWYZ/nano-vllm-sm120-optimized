import queue
import threading
import unittest
from types import SimpleNamespace

from nanovllm.sampling_params import SamplingParams
from nanovllm.server import GenerationJob, InferenceWorker


class TestInferenceAdmission(unittest.TestCase):
    def setUp(self):
        self.worker = InferenceWorker.__new__(InferenceWorker)
        self.worker.llm = SimpleNamespace(
            scheduler=SimpleNamespace(max_num_seqs=1)
        )
        self.worker.incoming = queue.Queue(maxsize=1)
        self.worker.capacity = threading.BoundedSemaphore(2)
        self.worker.failure = None
        self.worker.stop_event = threading.Event()

    @staticmethod
    def job():
        return GenerationJob([1, 2], SamplingParams(max_tokens=2))

    def test_total_capacity_rejects_overload_and_returns_slot(self):
        first = self.job()
        self.worker.submit(first)
        with self.assertRaises(queue.Full):
            self.worker.submit(self.job())

        self.assertIs(self.worker.incoming.get_nowait(), first)
        self.worker._finish(first, "stop")
        second = self.job()
        self.worker.submit(second)
        self.assertIs(self.worker.incoming.get_nowait(), second)

    def test_invalid_generation_limits_fail_early(self):
        with self.assertRaises(ValueError):
            SamplingParams(temperature=float("nan"))
        with self.assertRaises(ValueError):
            SamplingParams(max_tokens=0)


if __name__ == "__main__":
    unittest.main()
