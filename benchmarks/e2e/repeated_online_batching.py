"""Repeat mixed vs prefill-first online-arrival latency in alternating processes."""
import argparse, json, statistics, subprocess, sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--runs",type=int,default=5)
    args=parser.parse_args()
    root=Path(args.output_dir); root.mkdir(parents=True,exist_ok=True)
    results={"mixed":[],"prefill-first":[]}
    order=[policy for i in range(args.runs) for policy in
           (("mixed","prefill-first") if i%2==0 else ("prefill-first","mixed"))]
    for index,policy in enumerate(order,1):
        path=root/f"{index:02d}_{policy}.json"
        subprocess.run([sys.executable,"-m","benchmarks.e2e.online_batching",
                        "--model",args.model,"--policy",policy,"--output",str(path)],check=True)
        results[policy].append(json.loads(path.read_text()))
    summary={"method":"alternating_independent_processes","runs_per_policy":args.runs,
             "results":{}}
    for policy,items in results.items():
        keys=("mean_inter_token_gap_seconds","p95_inter_token_gap_seconds",
              "max_inter_token_gap_seconds")
        summary["results"][policy]={}
        for key in keys:
            values=[item["short_decode"][key] for item in items]
            summary["results"][policy][key]={
                "median":statistics.median(values),"mean":statistics.mean(values),
                "stdev":statistics.stdev(values),"min":min(values),"max":max(values)}
        summary["results"][policy]["raw_files"]=[
            f"{i:02d}_{policy}.json" for i in range(1,len(order)+1)
            if (root/f"{i:02d}_{policy}.json").exists()]
        print(f'{policy}: p95 median={summary["results"][policy]["p95_inter_token_gap_seconds"]["median"]*1000:.2f} ms, '
              f'max median={summary["results"][policy]["max_inter_token_gap_seconds"]["median"]*1000:.2f} ms')
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(f"summary={root/'summary.json'}")

if __name__=="__main__":
    main()
