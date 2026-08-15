import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def config(enable_mixed_batching=True, chunk_size=32):
    return SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        eos=-1,
        kvcache_block_size=16,
        num_kvcache_blocks=64,
        max_prefill_chunk_size=chunk_size,
        prefix_cache_policy="lru",
        enable_mixed_batching=enable_mixed_batching,
    )


class TestMixedScheduler(unittest.TestCase):
    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 16

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    @staticmethod
    def add_running(scheduler, length=20):
        seq = Sequence(list(range(length)))
        scheduler.block_manager.allocate(seq, 0)
        seq.num_cached_tokens = length - 1
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        scheduler.running.append(seq)
        return seq

    def test_decode_and_chunked_prefill_share_a_batch(self):
        scheduler = Scheduler(config())
        running = self.add_running(scheduler)
        waiting = Sequence(list(range(100, 180)))
        scheduler.add(waiting)

        scheduled, uses_prefill_path = scheduler.schedule()

        self.assertTrue(uses_prefill_path)
        self.assertEqual(scheduled, [running, waiting])
        self.assertEqual(running.num_scheduled_tokens, 1)
        self.assertEqual(waiting.num_scheduled_tokens, 32)
        self.assertEqual(
            scheduler.last_batch_stats,
            {"prefill_tokens": 32, "decode_tokens": 1},
        )
        self.assertEqual(scheduler.metrics["mixed_batches"], 1)

        scheduler.postprocess(scheduled, [999, 888], uses_prefill_path)
        self.assertEqual(running.last_token, 999)
        self.assertEqual(waiting.num_cached_tokens, 32)
        self.assertEqual(waiting.num_completion_tokens, 0)

    def test_ablation_policy_stalls_decode_for_prefill(self):
        scheduler = Scheduler(config(enable_mixed_batching=False))
        running = self.add_running(scheduler)
        waiting = Sequence(list(range(100, 180)))
        scheduler.add(waiting)

        scheduled, uses_prefill_path = scheduler.schedule()

        self.assertTrue(uses_prefill_path)
        self.assertEqual(scheduled, [waiting])
        self.assertNotIn(running, scheduled)
        self.assertEqual(scheduler.metrics["prefill_only_batches"], 1)

    def test_chunk_cap_rotates_across_waiting_requests(self):
        scheduler = Scheduler(config(chunk_size=16))
        first = Sequence(list(range(80)))
        second = Sequence(list(range(100, 180)))
        scheduler.add(first)
        scheduler.add(second)

        scheduled, uses_prefill_path = scheduler.schedule()

        self.assertTrue(uses_prefill_path)
        self.assertEqual(scheduled, [first, second])
        self.assertEqual(first.num_scheduled_tokens, 16)
        self.assertEqual(second.num_scheduled_tokens, 16)
        self.assertEqual(list(scheduler.waiting), [first, second])
        self.assertEqual(scheduler.metrics["prefill_chunks"], 2)

    def test_decode_candidate_cannot_preempt_already_scheduled_sequence(self):
        constrained = config()
        constrained.num_kvcache_blocks = 4
        scheduler = Scheduler(constrained)
        first = self.add_running(scheduler, length=20)
        second = self.add_running(scheduler, length=17)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 0)

        scheduled, uses_prefill_path = scheduler.schedule()

        self.assertTrue(uses_prefill_path)
        self.assertEqual(scheduled, [first, second])
        self.assertIn(first, scheduler.running)
        self.assertTrue(first.block_table)
        self.assertIn(second, scheduler.running)
        self.assertTrue(second.block_table)
        self.assertEqual(second.status, SequenceStatus.RUNNING)
        self.assertEqual(scheduler.metrics["preemptions"], 1)
        self.assertTrue(all(seq.block_table for seq in scheduled))


if __name__ == "__main__":
    unittest.main()
