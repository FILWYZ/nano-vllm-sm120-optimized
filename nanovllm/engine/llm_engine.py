import atexit
from dataclasses import fields
from time import perf_counter
from typing import Any
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.last_metrics: dict[str, Any] = {}
        atexit.register(self.exit)

    def exit(self):
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            return
        model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_seconds = decode_seconds = 0.0
        prefill_tokens = decode_tokens = 0
        prefill_iterations = decode_iterations = 0
        started = perf_counter()
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                elapsed = perf_counter() - t
                prefill_seconds += elapsed
                prefill_tokens += num_tokens
                prefill_iterations += 1
                prefill_throughput = num_tokens / elapsed
            else:
                elapsed = perf_counter() - t
                decode_seconds += elapsed
                decode_tokens -= num_tokens
                decode_iterations += 1
                decode_throughput = -num_tokens / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        total_seconds = perf_counter() - started
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        self.last_metrics = {
            "requests": len(prompts),
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "output_tokens": output_tokens,
            "prefill_iterations": prefill_iterations,
            "decode_iterations": decode_iterations,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "prefill_tokens_per_second": prefill_tokens / prefill_seconds if prefill_seconds else 0.0,
            "decode_tokens_per_second": decode_tokens / decode_seconds if decode_seconds else 0.0,
            "output_tokens_per_second": output_tokens / total_seconds if total_seconds else 0.0,
            "batch_ttft_seconds": prefill_seconds,
            "mean_tpot_seconds": decode_seconds / decode_iterations if decode_iterations else 0.0,
        }
        return outputs
