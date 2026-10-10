# Final commit validation — 2026-10-10

Tested source commit: `b365c6b3ff6657c5ba088626c64a3a59c2fda0a7` (`docs: publish final SM120 performance evidence`). The test ran from a clean detached checkout on RTX 5060 Laptop, SM120, driver 610.74, PyTorch 2.14.1+cu130, CUDA Runtime 13.0, FlashInfer 0.6.6, and Triton 3.8.0.

Command: `python -m pytest -q`

Result: **45 passed, 5 subtests passed**, with 6 FlashInfer/CUTLASS deprecation warnings. This is the full nano-vLLM test suite from that commit. The suite was run again after rebuilding the SM120 extension; it produced the same result.

This is a test record for the listed environment, not a claim about other GPU/software configurations. The documentation commit did not change nano-vLLM runtime code.
