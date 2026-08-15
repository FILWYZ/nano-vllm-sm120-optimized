from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.max_prefill_chunk_size = config.max_prefill_chunk_size
        self.enable_mixed_batching = config.enable_mixed_batching
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.metrics = {
            "prefill_chunks": 0,
            "mixed_batches": 0,
            "prefill_only_batches": 0,
            "decode_only_batches": 0,
            "preemptions": 0,
        }
        self.last_batch_stats = {"prefill_tokens": 0, "decode_tokens": 0}

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _schedule_decode(self, scheduled_seqs):
        decode_tokens = 0
        candidates = len(self.running)
        for _ in range(candidates):
            if len(scheduled_seqs) >= self.max_num_seqs:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq) and self.running:
                self.preempt(self.running.pop())
            if not self.block_manager.can_append(seq):
                self.preempt(seq)
                continue
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
            self.running.append(seq)
            decode_tokens += 1
        return decode_tokens

    def _schedule_prefill(self, scheduled_seqs, used_tokens):
        prefill_tokens = 0
        candidates = len(self.waiting)
        for _ in range(candidates):
            if len(scheduled_seqs) >= self.max_num_seqs:
                break
            remaining = self.max_num_batched_tokens - used_tokens - prefill_tokens
            if remaining <= 0:
                break
            seq = self.waiting.popleft()
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    self.waiting.append(seq)
                    continue
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_cached_blocks = 0
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            chunk_size = min(
                num_tokens, remaining, self.max_prefill_chunk_size
            )
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = chunk_size
            prefill_tokens += chunk_size
            self.metrics["prefill_chunks"] += 1
            if seq.num_cached_tokens + chunk_size == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                self.waiting.append(seq)
            scheduled_seqs.append(seq)
        return prefill_tokens

    def _record_batch(self, prefill_tokens, decode_tokens):
        self.last_batch_stats = {
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
        }
        if prefill_tokens and decode_tokens:
            self.metrics["mixed_batches"] += 1
        elif prefill_tokens:
            self.metrics["prefill_only_batches"] += 1
        else:
            self.metrics["decode_only_batches"] += 1

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        prefill_tokens = decode_tokens = 0
        if self.enable_mixed_batching:
            decode_tokens = self._schedule_decode(scheduled_seqs)
            prefill_tokens = self._schedule_prefill(
                scheduled_seqs, decode_tokens
            )
        else:
            prefill_tokens = self._schedule_prefill(scheduled_seqs, 0)
            if not prefill_tokens:
                decode_tokens = self._schedule_decode(scheduled_seqs)
        assert scheduled_seqs
        self._record_batch(prefill_tokens, decode_tokens)
        return scheduled_seqs, bool(prefill_tokens)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self.metrics["preemptions"] += 1

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
