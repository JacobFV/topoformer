"""Extended-04 budget ledger, computed from ALL job-wrapper receipts on the pro6000.

Main jobs: ~/tensegra-campaign04/results/<job>-process; dev/test/audit work:
~/tensegra-campaign04/results/dev/<tag>-<utc>-<pid>-process (bin/metered.sh).
CPU = sum of process-tree core-seconds over every receipt (failed/capped included).
GPU = device occupancy: union of wall intervals of main jobs, plus dev jobs whose tag
starts with "gpu" (CPU-only dev work runs with CUDA_VISIBLE_DEVICES empty).
Writes research/results/campaign-04/receipts.json and updates budget.json charged fields.
"""
import json, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUDGET = REPO / "research/campaigns/extended-04/budget.json"
OUT = REPO / "research/results/campaign-04/receipts.json"
SCRIPT = r'''
import json, glob, os
rows = []
for occ in glob.glob(os.path.expanduser("~/tensegra-campaign04/results/**/*-process/occupancy.json"), recursive=True):
    d = os.path.dirname(occ)
    try:
        o = json.load(open(occ)); l = json.load(open(os.path.join(d, "launch.json")))
    except Exception:
        continue
    rows.append({"dir": d.split("/results/", 1)[1], "dev": "/dev/" in d, "started_unix": l["started_unix"],
                 "ended_unix": o["ended_unix"], "wall_seconds": o["wall_seconds"], "cpu_core_seconds": o["cpu_core_seconds"],
                 "exit_code": o["exit_code"], "stop_reason": o.get("stop_reason")})
running = [os.path.dirname(p) for p in glob.glob(os.path.expanduser("~/tensegra-campaign04/results/**/*-process/launch.json"), recursive=True)
           if not os.path.exists(os.path.join(os.path.dirname(p), "occupancy.json"))]
print(json.dumps({"rows": rows, "running": [r.split("/results/", 1)[1] for r in running]}))
'''


def main():
    out = subprocess.run(["ssh", "pro6000", "wsl -d Ubuntu -- python3 -"], input=SCRIPT.encode(), capture_output=True, check=True)
    data = json.loads(out.stdout.decode().replace("\0", ""))
    rows = sorted(data["rows"], key=lambda r: r["started_unix"])
    def gpu(r):
        return (not r["dev"]) or r["dir"].split("/")[-1].startswith("gpu")
    spans = sorted((r["started_unix"], r["ended_unix"]) for r in rows if gpu(r))
    union, cur = 0.0, None
    for s, e in spans:
        if cur is None or s > cur[1]:
            if cur: union += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur: union += cur[1] - cur[0]
    cpu = sum(r["cpu_core_seconds"] for r in rows)
    dev_cpu = sum(r["cpu_core_seconds"] for r in rows if r["dev"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "running": data["running"]}, indent=1) + "\n")
    b = json.loads(BUDGET.read_text())
    b["charged_cpu_core_seconds"] = round(cpu, 1); b["charged_gpu_seconds"] = round(union, 1)
    b["charged_breakdown"] = {"main_cpu": round(cpu - dev_cpu, 1), "dev_cpu": round(dev_cpu, 1), "receipts": len(rows),
                              "running_unreceipted": data["running"]}
    b["jobs"] = "see research/results/campaign-04/receipts.json"
    BUDGET.write_text(json.dumps(b, indent=2) + "\n")
    c = b["ceilings"]
    print(f"cpu {cpu:.0f}/{c['cpu_core_seconds']} (dev {dev_cpu:.0f}), gpu {union:.0f}/{c['gpu_device_seconds']}, "
          f"receipts {len(rows)}, running {len(data['running'])}")


if __name__ == "__main__":
    main()
