import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    gpu_memory_reserve_gib: float = 0.0
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    attention_backend: str = "auto"
    rmsnorm_backend: str = "torch"
    qk_norm_rope_backend: str = "torch"
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    max_prefill_chunk_size: int = 512
    enable_mixed_batching: bool = True
    prefix_cache_policy: str = "lru"
    reserve_decode_kv: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert 16 <= self.kvcache_block_size <= 256
        assert self.kvcache_block_size & (self.kvcache_block_size - 1) == 0
        if self.attention_backend == "flash":
            assert self.kvcache_block_size == 256
        assert 1 <= self.tensor_parallel_size <= 8
        if not 0.1 <= self.gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be in [0.1, 1.0)")
        if self.gpu_memory_reserve_gib < 0:
            raise ValueError("gpu_memory_reserve_gib cannot be negative")
        assert self.attention_backend in {"auto", "flash", "flashinfer", "sdpa"}
        assert self.rmsnorm_backend in {"auto", "torch", "sm120"}
        assert self.qk_norm_rope_backend in {"auto", "torch", "sm120"}
        self.hf_config = AutoConfig.from_pretrained(self.model)
        assert self.max_prefill_chunk_size > 0
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.prefix_cache_policy in {"lru", "fifo"}
