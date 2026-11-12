"""Modal budget guard. Free tier = $30/month of credit per workspace; anything above is billed.
  python budget_guard.py check PROFILE      -> exit 0 if launching is OK, 1 if the workspace is at/over its cap (prints spend)
  python budget_guard.py watch PROFILE...   -> every 60 s: any workspace at/over its cap gets every live app stopped (modal app stop -y)
Cap: BUDGET_CAP in $ of this month's metered cost (default 28.5 = $30 credit minus a $1.50 margin; billing lags a few minutes)."""
import json, os, re, subprocess, sys, time
import os
CAP = float(os.environ.get("BUDGET_CAP", 30.0 - 1.5))  # metered $ this month at which to stop; set it per workspace
M = ["uv", "run", "-q", "--python", "3.12", "--with", "modal==1.6.3", "modal"]

def run(profile, *args):
    return subprocess.run(M + list(args), env={**os.environ, "MODAL_PROFILE": profile}, capture_output=True, text=True, timeout=120).stdout

def spend(profile):
    m = re.search(r"Metered Cost:\s+([\d.]+)", run(profile, "billing", "summary"))
    return float(m.group(1)) if m else None

def stop_all(profile):
    apps = json.loads(run(profile, "app", "list", "--json") or "[]")
    live = [a for a in apps if a.get("state") not in ("stopped",)]
    for a in live:
        run(profile, "app", "stop", "-y", a["app_id"])
    return [(a["app_id"], a["description"], a["state"]) for a in live]

if __name__ == "__main__":
    cmd, profiles = sys.argv[1], sys.argv[2:]
    if cmd == "check":
        s = spend(profiles[0]); ok = s is not None and s < CAP
        print(f"{profiles[0]}: metered ${s} of ${CAP:.2f} cap -> {'OK to launch' if ok else 'OVER BUDGET: do not launch'}"); sys.exit(0 if ok else 1)
    while True:
        for p in profiles:
            s = spend(p)
            if s is None or s >= CAP:
                stopped = stop_all(p)
                if stopped:
                    print(time.strftime("%H:%M:%S"), f"BUDGET STOP {p} (${s}): stopped {stopped}", flush=True)
        time.sleep(60)
