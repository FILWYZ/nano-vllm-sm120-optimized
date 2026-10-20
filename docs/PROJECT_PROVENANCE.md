# Project provenance and dependency record

- The original WSL project was extracted from a GitHub ZIP and had no .git directory.
- A local Git repository was initialized on 2026-10-07. Its first commit preserves the imported archive as an immutable local baseline. This commit is not an upstream commit.
- The upstream URL is recorded as origin: https://github.com/FILWYZ/nano-vllm-sm120-optimized.git
- GitHub advertised 29b87882c2898ffab081103346f6070267213ca7 as HEAD when queried. A full shallow clone could not be fetched from WSL because the HTTPS connection timed out, so the ZIP contents were not byte-compared against that commit.
- requirements-lock.txt records the tested third-party environment. The optional SM120 kernel package was built from temporary source and is omitted from the lock; its PyTorch fallbacks remain available.
