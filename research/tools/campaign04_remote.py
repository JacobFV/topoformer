"""Root-only helper for extended-04 jobs on the pro6000 (Windows host, WSL2 Ubuntu).

See research/campaigns/extended-04/infrastructure.md. Subcommands:
  snapshot                 push the current worktree HEAD (must be clean) as source-<sha>
  launch SHA CONFIG...     launch each config as a detached job-wrapped systemd unit
  launch-eval SHA CONFIG   launch a sealed evaluation (campaign02_evaluate.py, cuda) for one config
  launch-cmd JOB SHA -- CMD...  launch an arbitrary command (cwd = source snapshot) as a detached job-wrapped unit
  status                   list campaign units and finished receipts
  fetch JOB...             copy launch/occupancy/state receipts to research/results/campaign-04/
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = "/home/brand/tensegra-campaign04"
PY = "/home/brand/tensegra-campaign03/env/bin/python"
REPO = Path(__file__).resolve().parents[2]


def wsl(script: str) -> str:
    """Run a bash script in WSL via stdin (avoids cmd.exe quoting)."""
    out = subprocess.run(["ssh", "pro6000", "wsl -d Ubuntu -- bash -s"], input=script.encode(), capture_output=True, check=True)
    return out.stdout.decode(errors="replace").replace("\0", "")


def head_sha() -> str:
    if subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip():
        sys.exit("worktree not clean; commit before snapshotting")
    return subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def snapshot(_):
    sha = head_sha()
    tar = subprocess.Popen(["tar", "--exclude=.git", "--exclude=research/results", "-czf", "-", "."], cwd=REPO, stdout=subprocess.PIPE)
    dest = f"{ROOT}/source-{sha}"
    subprocess.run(["ssh", "pro6000", f'wsl -d Ubuntu -- bash -c "mkdir -p {dest} && tar -xzf - -C {dest}"'], stdin=tar.stdout, check=True)
    tar.wait()
    print(f"source-{sha}")


def launch(a):
    lines = []
    for cfg in a.configs:
        job = Path(cfg).stem
        lines.append(
            f"/home/brand/tensegra-campaign03/bin/detach.sh {job} {ROOT}/source-{a.sha} {PY} {ROOT}/bin/job.py "
            f"--output {ROOT}/results/{job}-process --wall-cap {a.wall_cap} --cpu-cap {a.cpu_cap} -- "
            f"{PY} -m tensegra.campaign02_population --config {cfg} --output {ROOT}/results/{job}")
    print(wsl("set -e\n" + "\n".join(lines) + "\n"))


def launch_eval(a):
    job = Path(a.config).stem
    print(wsl(f"/home/brand/tensegra-campaign03/bin/detach.sh {job} {ROOT}/source-{a.sha} {PY} {ROOT}/bin/job.py "
              f"--output {ROOT}/results/{job}-process --wall-cap {a.wall_cap} --cpu-cap {a.cpu_cap} -- "
              f"{PY} research/tools/campaign02_evaluate.py {a.config} --output {ROOT}/results/{job} --device cuda --threads 1\n"))


def launch_cmd(a):
    cmd = " ".join(a.cmd)
    print(wsl(f"/home/brand/tensegra-campaign03/bin/detach.sh {a.job} {ROOT}/source-{a.sha} {PY} {ROOT}/bin/job.py "
              f"--output {ROOT}/results/{a.job}-process --wall-cap {a.wall_cap} --cpu-cap {a.cpu_cap} -- {cmd}\n"))


def status(_):
    print(wsl(f"""systemctl --user list-units --type=service --no-legend --plain 'e04-*' 'a1-*' 'a2-*' 'b-*' 'c-*' 'f-*' | awk '{{print $1, $3, $4}}'
cd {ROOT}/results
for f in *-process/occupancy.json; do [ -e "$f" ] || continue
  python3 -c "import json,sys;o=json.load(open('$f'));print('$f'.split('/')[0], o['exit_code'], round(o['wall_seconds'],1), round(o['cpu_core_seconds'],1))"
done"""))


def fetch(a):
    dest = REPO / "research/results/campaign-04"
    dest.mkdir(parents=True, exist_ok=True)
    names = " ".join(f"{j}-process/launch.json {j}-process/occupancy.json" for j in a.jobs)
    tar = subprocess.run(["ssh", "pro6000", f'wsl -d Ubuntu -- bash -c "cd {ROOT}/results && tar -czf - {names}"'], capture_output=True, check=True)
    subprocess.run(["tar", "-xzf", "-", "-C", str(dest)], input=tar.stdout, check=True)
    for j in a.jobs:
        o = json.loads((dest / f"{j}-process/occupancy.json").read_text())
        print(j, o["exit_code"], o["wall_seconds"], o["cpu_core_seconds"])


def main():
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("snapshot").set_defaults(f=snapshot)
    l = s.add_parser("launch"); l.add_argument("sha"); l.add_argument("configs", nargs="+")
    l.add_argument("--wall-cap", type=float, default=5400); l.add_argument("--cpu-cap", type=float, default=7200); l.set_defaults(f=launch)
    e = s.add_parser("launch-eval"); e.add_argument("sha"); e.add_argument("config")
    e.add_argument("--wall-cap", type=float, default=14400); e.add_argument("--cpu-cap", type=float, default=28800); e.set_defaults(f=launch_eval)
    c = s.add_parser("launch-cmd"); c.add_argument("job"); c.add_argument("sha"); c.add_argument("cmd", nargs=argparse.REMAINDER)
    c.add_argument("--wall-cap", type=float, default=7200); c.add_argument("--cpu-cap", type=float, default=14400); c.set_defaults(f=launch_cmd)
    s.add_parser("status").set_defaults(f=status)
    f = s.add_parser("fetch"); f.add_argument("jobs", nargs="+"); f.set_defaults(f=fetch)
    a = p.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
