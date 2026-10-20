"""Small local OpenAI-compatible API with bounded admission and token streaming."""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch

from nanovllm import LLM, SamplingParams


@dataclass
class GenerationJob:
    prompt_ids: list[int]
    sampling: SamplingParams
    stop: list[str] = field(default_factory=list)
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=4096))
    cancelled: threading.Event = field(default_factory=threading.Event)
    seq: Any = None
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    sent_chars: int = 0
    finish_reason: str = "stop"
    error: str | None = None
    finished: bool = False
    slot_released: bool = False


class InferenceWorker:
    """Single owner of the engine; serializes CUDA calls and continuously batches work."""

    def __init__(self, llm: LLM, max_queue_size: int = 32):
        self.llm = llm
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self.incoming: queue.Queue[GenerationJob] = queue.Queue(maxsize=max_queue_size)
        self.capacity = threading.BoundedSemaphore(max_queue_size + self.max_active)
        self.pending: deque[GenerationJob] = deque()
        self.active: dict[int, GenerationJob] = {}
        self.stop_event = threading.Event()
        self.failure: str | None = None
        self.thread = threading.Thread(target=self._run, name="nano-vllm-worker", daemon=True)
        self.thread.start()

    @property
    def max_active(self) -> int:
        return self.llm.scheduler.max_num_seqs

    def submit(self, job: GenerationJob) -> None:
        if self.stop_event.is_set():
            raise RuntimeError("inference worker is shutting down")
        if self.failure:
            raise RuntimeError(self.failure)
        if not self.capacity.acquire(blocking=False):
            raise queue.Full
        try:
            self.incoming.put_nowait(job)
        except queue.Full:
            self.capacity.release()
            raise

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("inference worker did not stop within 30 seconds")

    def _emit(self, job: GenerationJob, event: tuple) -> None:
        try:
            job.events.put(event, timeout=0.25)
        except queue.Full:
            job.cancelled.set()

    def _finish(self, job: GenerationJob, reason: str = "stop", error: str | None = None) -> None:
        if job.finished:
            return
        job.finish_reason = reason
        job.error = error
        job.finished = True
        if not job.slot_released:
            job.slot_released = True
            self.capacity.release()
        self._emit(job, ("done",))

    def _drain_incoming(self, wait: bool = False) -> None:
        if wait and not self.pending:
            try:
                self.pending.append(self.incoming.get(timeout=0.1))
            except queue.Empty:
                return
        while True:
            try:
                self.pending.append(self.incoming.get_nowait())
            except queue.Empty:
                return

    def _admit(self) -> None:
        while self.pending and len(self.active) < self.max_active:
            job = self.pending.popleft()
            if job.cancelled.is_set() or self.stop_event.is_set():
                self._finish(job, "cancelled")
                continue
            try:
                seq_id = self.llm.add_request(job.prompt_ids, job.sampling)
                job.seq = self.llm.scheduler.waiting[-1]
                self.active[seq_id] = job
            except Exception as exc:
                self._finish(job, error=str(exc))

    def _cancel_active(self) -> None:
        for seq_id, job in list(self.active.items()):
            if job.cancelled.is_set() or self.stop_event.is_set():
                self.llm.scheduler.cancel(job.seq)
                self._finish(job, "cancelled")
                self.active.pop(seq_id, None)

    def _publish_text(self, job: GenerationJob, final: bool = False) -> bool:
        completion = job.seq.completion_token_ids
        job.token_ids = list(completion)
        job.text = self.llm.tokenizer.decode(
            completion, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        stop_positions = [job.text.find(stop) for stop in job.stop if stop and stop in job.text]
        if stop_positions:
            stop_at = min(stop_positions)
            safe_text = job.text[:stop_at]
            job.text = safe_text
            if len(safe_text) > job.sent_chars:
                self._emit(job, ("token", safe_text[job.sent_chars:]))
                job.sent_chars = len(safe_text)
            return True
        holdback = max((len(stop) for stop in job.stop), default=0)
        safe_end = len(job.text) if final or not holdback else max(0, len(job.text) - holdback + 1)
        if safe_end > job.sent_chars:
            self._emit(job, ("token", job.text[job.sent_chars:safe_end]))
            job.sent_chars = safe_end
        return False

    def _run(self) -> None:
        try:
            while (
                not self.stop_event.is_set()
                or self.active
                or self.pending
                or not self.incoming.empty()
            ):
                self._drain_incoming(
                    wait=not self.stop_event.is_set()
                    and not self.active and not self.pending
                )
                self._cancel_active()
                self._admit()
                if not self.active:
                    continue
                self.llm.step()
                for seq_id, job in list(self.active.items()):
                    stopped = self._publish_text(job)
                    if stopped:
                        self.llm.scheduler.cancel(job.seq)
                        self._publish_text(job, final=True)
                        self._finish(job, "stop")
                        self.active.pop(seq_id, None)
                    elif job.seq.is_finished:
                        self._publish_text(job, final=True)
                        reason = "length" if len(job.token_ids) >= job.sampling.max_tokens else "stop"
                        self._finish(job, reason)
                        self.active.pop(seq_id, None)
            while True:
                try:
                    self._finish(self.incoming.get_nowait(), "cancelled")
                except queue.Empty:
                    break
        except Exception as exc:
            self.failure = f"inference worker failed: {type(exc).__name__}: {exc}"
            for job in list(self.active.values()) + list(self.pending):
                if job.seq is not None:
                    try:
                        self.llm.scheduler.cancel(job.seq)
                    except Exception:
                        pass
                self._finish(job, error=self.failure)
            while True:
                try:
                    self._finish(self.incoming.get_nowait(), error=self.failure)
                except queue.Empty:
                    break


class LocalAPIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "nano-vllm-local/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[api] {self.address_string()} {fmt % args}")

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if size <= 0 or size > 1_048_576:
            raise ValueError("request body must be between 1 byte and 1 MiB")
        data = json.loads(self.rfile.read(size))
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def do_GET(self) -> None:
        if self.path in {"/health", "/healthz"}:
            worker = self.server.worker
            memory = None
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                memory = {
                    "free_gib": round(free / 2**30, 3),
                    "total_gib": round(total / 2**30, 3),
                    "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                    "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
                }
            self._json(200 if not worker.failure else 503, {
                "status": "ok" if not worker.failure else "error",
                "queue_depth": worker.incoming.qsize() + len(worker.pending),
                "active_requests": len(worker.active),
                "gpu_memory": memory,
                "error": worker.failure,
            })
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{
                "id": self.server.model_name, "object": "model",
                "owned_by": "local", "created": int(time.time()),
            }]})
        else:
            self._json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/v1/completions"}:
            self._json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        try:
            body = self._body()
            job, prompt_tokens = self._make_job(body, self.path)
            self.server.worker.submit(job)
        except queue.Full:
            self._json(429, {"error": {"message": "request queue is full", "type": "rate_limit_error"}})
            return
        except (ValueError, TypeError, KeyError) as exc:
            self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return
        except RuntimeError as exc:
            self._json(503, {"error": {"message": str(exc), "type": "server_error"}})
            return

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        model = str(body.get("model") or self.server.model_name)
        if body.get("stream", False):
            self._stream(job, request_id, model, self.path.endswith("chat/completions"))
        else:
            self._complete(job, request_id, model, prompt_tokens, self.path.endswith("chat/completions"))

    def _make_job(self, body: dict, path: str) -> tuple[GenerationJob, int]:
        tokenizer = self.server.worker.llm.tokenizer
        if path.endswith("chat/completions"):
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty array")
            if not hasattr(tokenizer, "apply_chat_template"):
                raise ValueError("tokenizer has no chat template")
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("prompt must be a non-empty string")
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt tokenized to an empty sequence")
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 128))
        temperature = body.get("temperature", 0.6)
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if max_tokens > self.server.max_output_tokens:
            raise ValueError(f"max_tokens exceeds server limit ({self.server.max_output_tokens})")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("temperature must be a non-negative number")
        if len(prompt_ids) + max_tokens > self.server.max_model_len:
            raise ValueError(
                f"prompt plus max_tokens exceeds max_model_len={self.server.max_model_len}"
            )
        stop = body.get("stop", [])
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or any(not isinstance(x, str) for x in stop):
            raise ValueError("stop must be a string or array of strings")
        params = SamplingParams(
            temperature=float(temperature), max_tokens=max_tokens, ignore_eos=False
        )
        return GenerationJob(prompt_ids, params, stop=stop), len(prompt_ids)

    def _wait_event(self, job: GenerationJob, timeout: float = 300.0):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                job.cancelled.set()
                raise TimeoutError("generation timed out")
            try:
                return job.events.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if self.server.worker.failure:
                    raise RuntimeError(self.server.worker.failure)

    def _complete(self, job: GenerationJob, request_id: str, model: str,
                  prompt_tokens: int, chat: bool) -> None:
        try:
            while True:
                event = self._wait_event(job)
                if event[0] == "done":
                    break
                if event[0] == "token":
                    pass
            if job.error:
                self._json(500, {"error": {"message": job.error, "type": "server_error"}})
                return
            completion_tokens = len(job.token_ids)
            content = job.text
            choice = ({"message": {"role": "assistant", "content": content},
                       "finish_reason": job.finish_reason}
                      if chat else {"text": content, "finish_reason": job.finish_reason})
            choice["index"] = 0
            self._json(200, {
                "id": request_id,
                "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()), "model": model, "choices": [choice],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })
        except (TimeoutError, RuntimeError) as exc:
            job.cancelled.set()
            self._json(503, {"error": {"message": str(exc), "type": "server_error"}})

    def _stream(self, job: GenerationJob, request_id: str, model: str, chat: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.wfile.write(b"data: " + data + b"\n\n")
            self.wfile.flush()

        base = {"id": request_id, "object": "chat.completion.chunk" if chat else "text_completion",
                "created": int(time.time()), "model": model}
        try:
            if chat:
                send({**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                           "finish_reason": None}]})
            while True:
                event = self._wait_event(job)
                if event[0] == "done":
                    break
                if event[0] == "token":
                    delta = {"content": event[1]} if chat else {"text": event[1]}
                    send({**base, "choices": [{"index": 0, "delta": delta,
                                               "finish_reason": None}]})
            if job.error:
                send({**base, "choices": [{"index": 0, "delta": {},
                                           "finish_reason": "error"}]})
            else:
                send({**base, "choices": [{"index": 0, "delta": {},
                                           "finish_reason": job.finish_reason}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, RuntimeError):
            job.cancelled.set()

    def log_error(self, fmt: str, *args: Any) -> None:
        self.log_message(fmt, *args)


class LocalAPIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(llm: LLM, host: str = "127.0.0.1", port: int = 8000,
          max_queue_size: int = 32, max_output_tokens: int = 1024,
          max_model_len: int = 2048) -> None:
    worker = InferenceWorker(llm, max_queue_size=max_queue_size)
    server = LocalAPIServer((host, port), LocalAPIHandler)
    server.worker = worker
    server.model_name = llm.model_runner.config.model.rsplit("/", 1)[-1]
    server.max_output_tokens = max_output_tokens
    server.max_model_len = min(
        max_model_len, llm.model_runner.config.max_model_len
    )
    try:
        print(f"nano-vLLM local API listening on http://{host}:{port}")
        print("Endpoints: /health, /v1/models, /v1/completions, /v1/chat/completions")
        server.serve_forever(poll_interval=0.5)
    finally:
        server.shutdown()
        server.server_close()
        worker.close()
        llm.exit()
