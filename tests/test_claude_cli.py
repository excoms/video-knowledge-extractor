"""The claude CLI provider, with the subprocess stubbed.

Written after a real run failed with "claude CLI returned nothing", which told
nobody anything. Each failure mode must now produce something actionable.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vke.providers import claude_cli as mod            # noqa: E402
from vke.providers.base import ProviderError           # noqa: E402


class Result:
    def __init__(self, code=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


def _provider(responses):
    """responses: list of Result, returned in order for successive calls."""
    calls = []
    p = mod.ClaudeCliProvider.__new__(mod.ClaudeCliProvider)
    p.model, p.exe = None, "/usr/local/bin/claude"
    seq = list(responses)

    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None):
        calls.append({"cmd": cmd, "input": input})
        return seq.pop(0)

    p._run = lambda prompt, fmt: fake_run(
        [p.exe, "-p", "--output-format", fmt], input=prompt)
    return p, calls


def test_prompt_goes_on_stdin_not_argv():
    """A transcript chunk is far too big to be a command-line argument."""
    p, calls = _provider([Result(0, "the answer")])
    out = p.complete("SYS", "U" * 40000)
    assert out == "the answer"
    assert calls[0]["input"].endswith("U" * 100), "prompt must be on stdin"
    assert not any(len(a) > 1000 for a in calls[0]["cmd"]), "prompt leaked into argv"


def test_json_envelope_is_unwrapped_not_returned_raw():
    p, _ = _provider([Result(0, '{"type":"result","result":"real answer"}')])
    assert p.complete("s", "u") == "real answer"


def test_empty_text_output_retries_as_json():
    p, calls = _provider([Result(0, ""), Result(0, '{"result":"second try"}')])
    assert p.complete("s", "u") == "second try"
    assert len(calls) == 2, "should retry once with the json format"


def test_total_silence_explains_itself():
    p, _ = _provider([Result(0, ""), Result(0, "")])
    try:
        p.complete("s", "u")
    except ProviderError as e:
        text = str(e)
        for expected in ("exit code", "stdout", "stderr", "--provider"):
            assert expected in text, f"diagnosis missing {expected!r}: {text}"
    else:
        raise AssertionError("silence should raise")


def test_signed_out_says_how_to_sign_in():
    p, _ = _provider([Result(1, "", "authentication_error: token expired")])
    try:
        p.complete("s", "u")
    except ProviderError as e:
        assert "sign in" in str(e).lower()
        assert "exit code: 1" in str(e)
    else:
        raise AssertionError("auth failure should raise")


def test_error_envelope_is_detected_even_on_exit_zero():
    p, _ = _provider([Result(0, '{"type":"result","subtype":"error_during_execution","is_error":true}')])
    try:
        p.complete("s", "u")
    except ProviderError as e:
        assert "error_during_execution" in str(e)
    else:
        raise AssertionError("an error envelope should raise")


def test_timeout_is_reported_with_advice():
    p, _ = _provider([])
    def boom(prompt, fmt):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=900)
    p._run = boom
    try:
        p.complete("s", "u")
    except ProviderError as e:
        assert "timed out" in str(e) and "anthropic" in str(e)
    else:
        raise AssertionError("timeout should raise")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  pass  {name}")
            except Exception as e:
                failures += 1
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{'all passed' if not failures else str(failures) + ' failed'}")
    sys.exit(1 if failures else 0)
