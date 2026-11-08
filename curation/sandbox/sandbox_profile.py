"""The macOS sandbox that model-written code runs in on a local machine: no network, no writes outside its work dir, no exec but the
interpreter, no keychain lookups, no reads under $HOME but the interpreter's own install, and this repository's .env denied whatever else
is allowed. Error text that leaves the sandbox is scrubbed of key- or token-like strings.

Copied from hnaracodes/planningGemma, cloud-gemma/sera-e4b/leak/screen.py (the leakage screen), keeping only its sandbox; the .env it
protects is now this repository's. Used by lower_filter.py.
"""
import functools, json, os, pathlib, re, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]  # the repository: oracle/sandbox/ -> oracle/ -> root

SANDBOX_EXEC = "/usr/bin/sandbox-exec"  # absolute, so nothing on PATH can stand in for it


MIN_PATH = "/usr/bin:/bin"


ERR_MAX = 200


SECRETISH = [re.compile(r"(?i)\w*(key|token|secret|passw\w*|auth\w*|credential\w*)\w*\s*[=:]\s*\S+"),
             re.compile(r"(?=[\w+/=-]*\d)(?=[\w+/=-]*[A-Za-z])[\w+/=-]{16,}")]  # long letter+digit runs: API keys, tokens


def _require_sandbox():
    if not os.path.exists(SANDBOX_EXEC):
        raise SystemExit(f"This runs model-written code and needs {SANDBOX_EXEC} (macOS); none here, so nothing was run. "
                         "Run it on a Mac.")



def _sb(path):
    if not path.startswith("/") or any(c in path for c in '"\\\n'):
        raise SystemExit(f"path not usable in a sandbox profile: {path!r}")
    return f'"{path}"'



@functools.lru_cache(maxsize=None)
def _python_paths(py):
    """The interpreter's own install (venv and base), which the child must read even when it lies under $HOME."""
    code = "import json, os, sys; print(json.dumps([sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.realpath(sys.executable)]))"
    out = subprocess.run([py, "-I", "-c", code], capture_output=True, text=True, timeout=60, env={"PATH": MIN_PATH})
    if out.returncode:
        raise SystemExit(f"{py} does not start: {out.stderr[-300:]}")
    *dirs, exe = json.loads(out.stdout)
    return tuple(sorted({q for p in [*dirs, os.path.dirname(exe)] for q in (p, os.path.realpath(p))}))



@functools.lru_cache(maxsize=None)
def _env_files():
    """This repository's .env and, from a worktree, the main checkout's: denied even if the repo is outside $HOME."""
    out = {str(ROOT / ".env")}
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                         capture_output=True, text=True, timeout=30)
    if git.returncode == 0:  # not a git checkout: there is no main checkout to protect
        out.add(str(pathlib.Path(git.stdout.strip()).parent / ".env"))
    return tuple(sorted({q for p in out for q in (p, os.path.realpath(p))}))



def _profile(home, workdir, py):
    # Later rules win: deny $HOME, allow back the interpreter and the work dir, then deny the .env files whatever was allowed.
    # Python's startup realpath() stats each directory above a venv under $HOME: those get metadata only (no contents, no listing).
    envs = " ".join(f"(literal {_sb(p)})" for p in _env_files())
    reads = (workdir, *_python_paths(py))
    above = sorted({str(a) for p in reads for a in pathlib.PurePath(p).parents})
    # No exec (a trace never starts a program; gh or security could read the keychain) except the interpreter sandbox-exec starts,
    # and no keychain service lookups (Lebron James, 6704455358).
    real = os.path.realpath(py)
    app = pathlib.Path(real).parent.parent / "Resources/Python.app/Contents/MacOS/Python"  # framework Python re-execs into this
    execs = " ".join(f"(literal {_sb(q)})" for q in sorted({py, real} | ({str(app), os.path.realpath(app)} if app.exists() else set())))
    return "\n".join(["(version 1)", "(allow default)", "(deny network*)", "(deny file-write*)",
                      "(deny process-exec*)", f"(allow process-exec {execs})",
                      '(deny mach-lookup (global-name "com.apple.SecurityServer") (global-name "com.apple.securityd.xpc"))',
                      f'(allow file-write* (subpath {_sb(workdir)}) (literal "/dev/null"))',
                      f"(deny file-read* (subpath {_sb(home)}) {envs})",
                      f"(allow file-read-metadata {' '.join(f'(literal {_sb(a)})' for a in above)})",
                      f"(allow file-read* {' '.join(f'(subpath {_sb(p)})' for p in reads)})",
                      f"(deny file-read* {envs})"])



def scrub(err):
    """`Type: message` from a child, with key- or token-like text removed, at most ERR_MAX characters: it goes into a pushed report."""
    kind, _, msg = err.partition(": ")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,63}", kind):
        kind, msg = "Error", err
    for rx in SECRETISH:
        msg = rx.sub("<redacted>", msg)
    return f"{kind}: {msg}"[:ERR_MAX]

