"""The gemini constructor backend with a fake client (no network, no google-genai needed), backend selection by env, and the
--reuse-raw guard that keeps a Gemini run from picking up cached Claude responses for the same key."""
import json, os, pathlib, sys, tempfile, types, unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import constructor  # noqa: E402
from constructor import Request, Response, save_raw  # noqa: E402
from constructor.gemini_api import GeminiAPI  # noqa: E402
from port_kernelbench import saved_response  # noqa: E402

OUT = {"plan": "one block", "code": "def workload(x):\n    return x\n"}


def reply(text, finish="STOP"):
    return types.SimpleNamespace(text=text, model_version="gemini-test", candidates=[types.SimpleNamespace(finish_reason=finish)],
                                 usage_metadata=types.SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=7))


class FakeClient:
    def __init__(self, *results):
        self.results, self.calls = list(results), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class Gemini(unittest.TestCase):
    def test_parses_structured_output_and_usage(self):
        c = FakeClient(reply(json.dumps(OUT)))
        r = GeminiAPI(client=c).generate([Request("k1", "sys", "user")])[0]
        self.assertTrue(r.ok)
        self.assertEqual(r.output, OUT)
        self.assertEqual((r.backend, r.model), ("gemini", "gemini-test"))
        self.assertEqual(r.usage["output_tokens"], 12)  # answer + thinking
        cfg = c.calls[0][2]
        self.assertEqual(cfg["system_instruction"], "sys")
        self.assertEqual(cfg["response_json_schema"], constructor.DEFAULT_SCHEMA)
        self.assertGreaterEqual(cfg["max_output_tokens"], 65536)
        self.assertEqual(json.loads(r.raw)["structured_output"], OUT)  # the shape saved_response reads

    def test_missing_required_field_is_not_ok(self):
        r = GeminiAPI(client=FakeClient(reply(json.dumps({"plan": "x"}), finish="MAX_TOKENS"))).generate([Request("k", "s", "u")])[0]
        self.assertFalse(r.ok)
        self.assertIn("MAX_TOKENS", r.error)

    @mock.patch("constructor.gemini_api.time.sleep")
    def test_retries_429_then_succeeds(self, _sleep):
        c = FakeClient(ApiError(429), reply(json.dumps(OUT)))
        self.assertTrue(GeminiAPI(client=c).generate([Request("k", "s", "u")])[0].ok)
        self.assertEqual(len(c.calls), 2)

    @mock.patch("constructor.gemini_api.time.sleep")
    def test_does_not_retry_400(self, _sleep):
        c = FakeClient(ApiError(400), reply(json.dumps(OUT)))
        r = GeminiAPI(client=c).generate([Request("k", "s", "u")])[0]
        self.assertFalse(r.ok)
        self.assertEqual(len(c.calls), 1)

    def test_effort_maps_to_thinking_level(self):
        c = FakeClient(reply(json.dumps(OUT)))
        GeminiAPI(client=c, effort="low").generate([Request("k", "s", "u")])
        self.assertEqual(c.calls[0][2]["thinking_config"], {"thinking_level": "low"})


class Selection(unittest.TestCase):
    def test_env_picks_backend_and_model_when_flags_unset(self):
        with mock.patch.dict(os.environ, {"CONSTRUCTOR_BACKEND": "gemini", "CONSTRUCTOR_MODEL": "gemini-x"}), \
             mock.patch("constructor.gemini_api.GeminiAPI.__init__", return_value=None) as init:
            con = constructor.get_constructor(None, None, workers=2)
        self.assertIsInstance(con, GeminiAPI)
        init.assert_called_once_with(model="gemini-x", workers=2)

    def test_default_is_still_claude_cli(self):
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("CONSTRUCTOR_BACKEND", None)
            self.assertEqual(constructor.get_constructor(None, None).name, "claude_cli")


class ReuseGuard(unittest.TestCase):
    def test_saved_response_only_reuses_same_backend(self):
        with tempfile.TemporaryDirectory() as d:
            raw = pathlib.Path(d)
            claude = {"structured_output": {"plan": "claude", "code": "c"}, "usage": {}}
            save_raw(raw, Response("p1", True, claude["structured_output"], "claude-x", "claude_cli", {}, json.dumps(claude)))
            self.assertIsNone(saved_response(raw, "p1", "gemini"))  # a Gemini run must not pick up the Claude answer
            self.assertEqual(saved_response(raw, "p1", "claude_cli").output["plan"], "claude")
            gem = GeminiAPI(client=FakeClient(reply(json.dumps(OUT)))).generate([Request("p1", "s", "u")])[0]
            save_raw(raw, gem)
            got = saved_response(raw, "p1", "gemini")
            self.assertEqual((got.output, got.backend), (OUT, "gemini"))


if __name__ == "__main__":
    unittest.main()
