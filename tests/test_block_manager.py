import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class TestVariableBlockSize(unittest.TestCase):
    def setUp(self):
        self.original_block_size = Sequence.block_size

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    def test_allocation_and_prefix_reuse_at_supported_sizes(self):
        for block_size in (16, 32, 64, 128, 256):
            with self.subTest(block_size=block_size):
                Sequence.block_size = block_size
                manager = BlockManager(8, block_size)
                tokens = list(range(block_size * 2))
                first = Sequence(tokens)
                self.assertEqual(first.num_blocks, 2)
                self.assertEqual(manager.can_allocate(first), 0)
                manager.allocate(first, 0)
                first.num_scheduled_tokens = len(tokens)
                manager.hash_blocks(first)
                cached_id = first.block_table[0]
                manager.deallocate(first)

                second = Sequence(tokens)
                cached_blocks = manager.can_allocate(second)
                self.assertEqual(cached_blocks, 1)
                manager.allocate(second, cached_blocks)
                self.assertEqual(second.num_cached_tokens, block_size)
                self.assertEqual(second.block_table[0], cached_id)

    def test_append_allocates_only_at_page_boundary(self):
        Sequence.block_size = 16
        manager = BlockManager(4, 16)
        seq = Sequence(list(range(16)))
        manager.allocate(seq, 0)
        free_before = len(manager.free_block_ids)
        seq.append_token(16)
        self.assertTrue(manager.can_append(seq))
        manager.may_append(seq)
        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(len(manager.free_block_ids), free_before - 1)

    def test_decode_reservation_prevents_boundary_allocation(self):
        Sequence.block_size = 16
        manager = BlockManager(4, 16, reserve_decode_kv=True)
        seq = Sequence(list(range(16)))
        seq.max_tokens = 32
        self.assertEqual(manager.can_allocate(seq), 0)
        manager.allocate(seq, 0)
        self.assertEqual(len(seq.block_table), 3)
        self.assertEqual(len(manager.free_block_ids), 1)

        for token in range(16, 33):
            seq.append_token(token)
            self.assertTrue(manager.can_append(seq))
            manager.may_append(seq)
        self.assertEqual(len(seq.block_table), 3)
        self.assertEqual(len(manager.free_block_ids), 1)

        too_large = Sequence(list(range(16)))
        too_large.max_tokens = 64
        self.assertEqual(manager.can_allocate(too_large), -1)


if __name__ == "__main__":
    unittest.main()
