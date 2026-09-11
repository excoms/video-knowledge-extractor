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
    p.candidates, p._tried = ["/usr/local/bin/claude"], set()
    seq = list(responses)

    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None):
        calls.append({"cmd": cmd, "input": input})
        return seq.pop(0)

    p._run = lambda prompt, fmt: fake_run(
        [p.exe, "-p", "--output-format", fmt], input=prompt)
    return p, calls


def test_prompt_is_passed_as_an_argument_not_piped():
    """`claude -p <prompt>`. Piping the prompt instead yields a *successful*
    call with an empty result and ~3 input tokens — the prompt never arrives
    and nothing reports an error. Stubs subprocess.run so the real command
    construction is exercised, not a stand-in for it."""
    seen = {}

    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None):
        seen["cmd"], seen["input"] = cmd, input
        return Result(0, "the answer")

    real = mod.subprocess.run
    mod.subprocess.run = fake_run
    p = mod.ClaudeCliProvider.__new__(mod.ClaudeCliProvider)
    p.model, p.exe = None, "/usr/local/bin/claude"
    p.candidates, p._tried = ["/usr/local/bin/claude"], set()
    try:
        assert p.complete("SYS", "USER-BODY") == "the answer"
    finally:
        mod.subprocess.run = real

    assert seen["input"] is None, "prompt must not be piped"
    assert "USER-BODY" in " ".join(seen["cmd"]), "prompt must be an argument"
    assert seen["cmd"][1] == "-p"
    assert "--output-format" in seen["cmd"]


def test_empty_result_on_a_successful_call_is_an_error():
    """The exact failure seen in the wild: subtype success, result empty."""
    p, _ = _provider([Result(0, '{"type":"result","subtype":"success",'
                                '"is_error":false,"result":"",'
                                '"usage":{"input_tokens":3}}')])
    try:
        p.complete("s", "u")
    except ProviderError as e:
        assert "empty response" in str(e)
        assert "3 input tokens" in str(e), "should name the giveaway"
    else:
        raise AssertionError("an empty result should raise")


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


def test_falls_back_to_another_install_when_the_first_returns_empty():
    """Observed in the wild: npm-global 2.1.83 returns an empty result in -p
    mode while the native 2.1.81 beside it works. Move down the list rather
    than failing on whichever happens to be first on PATH."""
    empty = '{"type":"result","subtype":"success","result":"","usage":{"input_tokens":3}}'
    used = []

    p = mod.ClaudeCliProvider.__new__(mod.ClaudeCliProvider)
    p.model = None
    p.candidates = ["/usr/local/bin/claude", "/home/u/.local/bin/claude"]
    p.exe, p._tried = p.candidates[0], set()

    def fake_run(prompt, fmt):
        used.append(p.exe)
        if p.exe == p.candidates[0]:
            return Result(0, "" if fmt == "text" else empty)
        return Result(0, "the real answer")

    p._run = fake_run
    assert p.complete("s", "u") == "the real answer"
    assert p.candidates[1] in used, "should have tried the second install"


def test_reports_every_install_when_all_are_empty():
    empty = '{"type":"result","subtype":"success","result":"","usage":{"input_tokens":3}}'
    p = mod.ClaudeCliProvider.__new__(mod.ClaudeCliProvider)
    p.model = None
    p.candidates = ["/a/claude", "/b/claude"]
    p.exe, p._tried = p.candidates[0], set()
    p._run = lambda prompt, fmt: Result(0, "" if fmt == "text" else empty)
    try:
        p.complete("s", "u")
    except ProviderError as e:
        assert "/a/claude" in str(e) and "/b/claude" in str(e)
    else:
        raise AssertionError("should fail once every install is exhausted")


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
