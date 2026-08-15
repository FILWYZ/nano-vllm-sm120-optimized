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
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    attention_backend: str = "auto"
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    max_prefill_chunk_size: int = 512
    enable_mixed_batching: bool = True

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert 16 <= self.kvcache_block_size <= 256
        assert self.kvcache_block_size & (self.kvcache_block_size - 1) == 0
        if self.attention_backend == "flash":
            assert self.kvcache_block_size == 256
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.attention_backend in {"auto", "flash", "flashinfer", "sdpa"}
        self.hf_config = AutoConfig.from_pretrained(self.model)
        assert self.max_prefill_chunk_size > 0
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
