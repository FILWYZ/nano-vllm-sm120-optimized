import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class TestPrefixCachePolicy(unittest.TestCase):
    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 2

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    @staticmethod
    def seed_cached_block(manager, block_id, token_ids):
        block_hash = manager.compute_hash(token_ids)
        manager.blocks[block_id].update(block_hash, token_ids)
        manager.hash_to_block_id[block_hash] = block_id

    def test_lru_touch_changes_eviction_victim(self):
        lru = BlockManager(3, 2, "lru")
        self.seed_cached_block(lru, 0, [1, 2])
        self.seed_cached_block(lru, 1, [3, 4])
        seq = Sequence([1, 2, 9, 10])

        self.assertEqual(lru.can_allocate(seq), 1)
        self.assertEqual(list(lru.free_block_ids), [1, 2, 0])
        victim = lru._allocate_block()
        self.assertEqual(victim, 1)
        self.assertEqual(lru.metrics["block_hits"], 1)
        self.assertEqual(lru.metrics["token_hits"], 2)
        self.assertEqual(lru.metrics["cached_evictions"], 1)

    def test_fifo_does_not_touch_on_lookup(self):
        fifo = BlockManager(3, 2, "fifo")
        self.seed_cached_block(fifo, 0, [1, 2])
        self.seed_cached_block(fifo, 1, [3, 4])
        seq = Sequence([1, 2, 9, 10])

        self.assertEqual(fifo.can_allocate(seq), 1)
        self.assertEqual(list(fifo.free_block_ids), [0, 1, 2])
        self.assertEqual(fifo._allocate_block(), 0)

    def test_hash_collision_is_counted_and_rejected(self):
        manager = BlockManager(3, 2, "lru")
        requested = [1, 2]
        block_hash = manager.compute_hash(requested)
        manager.blocks[0].update(block_hash, [7, 8])
        manager.hash_to_block_id[block_hash] = 0

        cached = manager.can_allocate(Sequence(requested + [9, 10]))

        self.assertEqual(cached, 0)
        self.assertEqual(manager.metrics["hash_collisions"], 1)
        self.assertEqual(manager.metrics["misses"], 1)


if __name__ == "__main__":
    unittest.main()
