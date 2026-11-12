"""The constructor agent: the model that writes candidate kernels, ports KernelBench problems, authors p-style problems and wraps
scraped kernels. One protocol, one backend:

  claude_cli        `claude -p` headless under the user's Max plan (default)

Every request is a system prompt plus one user message and a JSON schema; every response is the parsed structured output plus usage.
Not called "teacher": that word belongs to other parts of the study.
"""
from __future__ import annotations

import dataclasses, json, pathlib, sys, typing

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import sha256  # noqa: E402

DEFAULT_SCHEMA = {"type": "object", "properties": {"plan": {"type": "string", "description": "3-6 lines: grid and block shapes, "
                  "accumulator dtype, what stays outside the kernel"}, "code": {"type": "string", "description": "the complete Python module"}},
                  "required": ["plan", "code"], "additionalProperties": False}


@dataclasses.dataclass
class Request:
    key: str                      # caller's id for this request (candidate_id, problem_id, ...)
    system: str
    user: str
    schema: dict = dataclasses.field(default_factory=lambda: DEFAULT_SCHEMA)
    max_tokens: int = 16000

    @property
    def prompt_sha256(self) -> str:
        return sha256(self.system + "\n\x00\n" + self.user)


@dataclasses.dataclass
class Response:
    key: str
    ok: bool
    output: dict | None           # parsed structured output (schema-shaped) or None
    model: str
    backend: str
    usage: dict                   # input_tokens, output_tokens, wall_s, cost_usd (when known)
    raw: str                      # the backend's full raw response text (saved to raw_responses/, not to records)
    error: str | None = None


class Constructor(typing.Protocol):
    name: str
    model: str

    def generate(self, requests: list) -> list: ...


def get_constructor(backend: str = "claude_cli", model: str | None = None, **kw) -> Constructor:
    if backend == "claude_cli":
        from constructor.claude_cli import ClaudeCLI
        return ClaudeCLI(model=model, **kw)
    raise ValueError(f"unknown constructor backend {backend!r}")


def save_raw(raw_dir: pathlib.Path, resp: Response) -> str:
    """Write the raw response under raw_dir/<sha>.json and return the sha (recorded on the candidate as raw_sha256)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    h = sha256(resp.raw)
    (raw_dir / f"{h}.json").write_text(json.dumps({"key": resp.key, "backend": resp.backend, "model": resp.model, "raw": resp.raw}))
    return h
