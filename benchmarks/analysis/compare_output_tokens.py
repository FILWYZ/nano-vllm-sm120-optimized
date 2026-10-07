import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark token arrays")
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()
    payloads = [json.loads(Path(path).read_text()) for path in args.results]
    reference = payloads[0]["output_tokens"]
    matches = [payload["output_tokens"] == reference for payload in payloads]
    total_tokens = sum(len(tokens) for tokens in reference)
    print(
        f"exact_match={all(matches)} files={len(matches)} "
        f"requests={len(reference)} tokens={total_tokens}"
    )
    if not all(matches):
        raise SystemExit("output token arrays differ")


if __name__ == "__main__":
    main()
