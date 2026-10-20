# Integrated candidate validation — 2026-10-20

## Tested source and environment

- Source commit: `11d0aa58a2c68d7132dd0d919e1029afea6a6aa2` (`integrate edge evidence and service paths`); tested from a clean detached checkout.
- RTX 5060 Laptop, compute capability 12.0, 8 GiB, WSL2 Ubuntu.
- NVIDIA driver reported by `nvidia-smi`: 610.57.01; CUDA Runtime 13.0; CUDA Toolkit/NVCC 13.3.73.
- Python 3.12.3; PyTorch 2.14.1+cu130; Triton 3.8.0; FlashInfer 0.6.6.

## Results

- `python -m pytest -q`: **47 passed**, 6 warnings, and **5 subtests passed** in 7.28 seconds.
- `python -m compileall`: passed for the project Python modules.
- `bash -n scripts/setup_wsl.sh`: passed.
- CLI entry-point smoke check: passed.
- Warnings were FlashInfer/CUTLASS deprecation notices; no test failures occurred.

This validation belongs to the integrated candidate commit above. It is distinct from the 2026-10-10 clean-commit validation (45 passed plus 5 subtests), and from the earlier PyTorch 2.11.0+cu128 / 23-test historical milestone documented in [`ENVIRONMENT.md`](ENVIRONMENT.md) and [`optimizations/ROADMAP_RESULTS.md`](optimizations/ROADMAP_RESULTS.md).

The three-round same-device compatibility-baseline comparison is reported separately in [`UPSTREAM_BASELINE_AUDIT_20261020.md`](UPSTREAM_BASELINE_AUDIT_20261020.md). A sustained power/thermal soak is reported separately in [`EDGE_SOAK_20261020.md`](EDGE_SOAK_20261020.md) after its data collection completes.
