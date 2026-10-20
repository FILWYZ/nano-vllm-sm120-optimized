# Project provenance and dependency record

## Repository history

- The earlier WSL working directory originated from an archive without Git metadata. Its locally initialized history is preserved for provenance; that history is not an upstream Git history.
- The current integration candidate is based on GitHub `main` commit `60c1399f3b8b5a8aa4edfd846c044e692c9c0a68` and is published from `codex/integrate-edge-evidence-20261020` for review.
- The former remote branch `codex/rtx5060-optimization-20261008` has no common ancestor with `main`. Its unique runtime, validation, and evidence files were inventoried and selectively transplanted onto the main-based integration candidate. The old branch is retained as an audit source until the integration is reviewed; it was not force-pushed or silently deleted.
- `baseline-upstream-2026-08-15` preserves the tested upstream source snapshot at `bb823b3e06983d71485a8e1f23715ebd87d98ef8`. `baseline-blackwell-v0` preserves the minimally adapted SM120 compatibility baseline at `7aa1f251cf1e041ded3371bab41f90e6736111ac`.
- The unmodified upstream source fails during local import because `flash_attn` is absent from the current environment. No direct-upstream performance score is claimed; the measured comparison is against the minimal compatibility baseline. See [`UPSTREAM_BASELINE_AUDIT_20261020.md`](UPSTREAM_BASELINE_AUDIT_20261020.md).

## Dependencies

- `requirements-lock-20261007-cu128.txt` is an early historical lock for the PyTorch 2.11.0+cu128 environment. It does not describe the final runtime.
- The current project metadata and install helper target PyTorch 2.14.x with CUDA 13.0 wheels. The validated environment used PyTorch 2.14.1+cu130, Triton 3.8.0 and FlashInfer 0.6.6; see [`ENVIRONMENT.md`](ENVIRONMENT.md).
- The optional SM120 extension lives in the separate `sm120-norm-fusion-kernels` repository. Framework fallbacks remain available when the extension is absent or the runtime shape is outside its measured dispatch range.
