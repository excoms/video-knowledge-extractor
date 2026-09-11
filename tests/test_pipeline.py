"""Offline checks for the parts that must not silently break.

No network, no model: a stub provider returns realistic output so these test
our code rather than someone else's service.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vke.analysis import analyse, chunk, locate_quote, plan_schema  # noqa: E402
from vke.normalise import clean_text                                # noqa: E402
from vke.transcripts.asr import (AsrUnavailable, available_backends,  # noqa: E402
                                 pick_backend, _installed)
from vke.transcripts import Segment, Transcript                     # noqa: E402

SAMPLE = ("واجب الوجود ایسے وجود کو کہا جاتا ہے جس کا موجود ہونا ایسا لابدی ہو "
          "کہ اس کے معدوم ہونے کا تصور بھی غلط ہو ممکن الوجود اسے کہا جاتا ہے "
          "جس کا ہونا یا نہ ہونا دونوں ممکنات میں سے ہوں")


class StubProvider:
    name = "stub"

    def complete(self, system, user, max_tokens=4000):
        if "design compact JSON schemas" in system:
            return ('```json\n{"record_name":"argument","fields":['
                    '{"name":"claim","description":"one sentence"},'
                    '{"name":"vulnerability","description":"weak point"}]}\n```')
        return json.dumps([{
            "fields": {"claim": "The universe is contingent",
                       "vulnerability": "Fallacy of composition"},
            "quote": "واجب الوجود ایسے وجود کو کہا جاتا ہے",
            "speaker": "Host", "confidence": "high"}], ensure_ascii=False)


def test_normalise_strips_non_speech_narration():
    assert clean_text("یہ ایک دلیل ہے موسیقی") == "یہ ایک دلیل ہے"
    assert clean_text("a claim [Music]") == "a claim"


def test_normalise_collapses_loops_but_keeps_real_repetition():
    assert clean_text("a b b b b b b b c") == "a b b b b c"
    assert clean_text("no doubt no doubt") == "no doubt no doubt"


def test_chunk_overlaps_and_never_truncates():
    text = ". ".join("word " * 400 for _ in range(6))
    parts = chunk(text, size=2000, overlap=200)
    assert len(parts) > 1
    assert sum(len(p) for p in parts) > len(text) * 0.99   # nothing dropped
    assert len(chunk("short text")) == 1


def test_locate_quote_matches_against_segments():
    segs = [Segment(i * 2.0, i * 2.0 + 2.0, w) for i, w in enumerate(SAMPLE.split())]
    ts = locate_quote(segs, "واجب الوجود ایسے وجود کو کہا جاتا ہے")
    assert ts is not None and ts[0] < ts[1]
    assert locate_quote(segs, "ab") is None                # too short to match


def test_plan_schema_reads_a_free_text_request():
    plan = plan_schema(StubProvider(), "Find every argument and its weakness")
    assert plan.record_name == "argument"
    assert [f["name"] for f in plan.fields] == ["claim", "vulnerability"]


def test_analyse_wraps_records_in_the_envelope():
    provider = StubProvider()
    plan = plan_schema(provider, "Find every argument")
    segs = [Segment(i * 2.0, i * 2.0 + 2.0, w) for i, w in enumerate(SAMPLE.split())]
    t = Transcript("vid1", "Test", "https://youtu.be/vid1", SAMPLE, "ur",
                   "asr:test", 60.0, segs)
    records = analyse(provider, t, plan, "Find every argument")
    assert len(records) == 1
    r = records[0]
    for key in ("schema_version", "video_id", "url", "speaker", "confidence",
                "quote_original", "timestamp", "record"):
        assert key in r, f"envelope missing {key}"
    assert r["video_id"] == "vid1"
    assert set(r["record"]) == {"claim", "vulnerability"}


def test_backend_detection_does_not_import_the_module():
    """find_spec, not import: mlx_whisper raises on import without a Metal
    device, which would make an installed backend look missing."""
    import sys
    sys.modules.pop("mlx_whisper", None)
    _installed("mlx_whisper")
    assert "mlx_whisper" not in sys.modules
    assert _installed("definitely_not_a_real_module_xyz") is False
    assert _installed("json") is True


def test_available_backends_reports_every_backend_with_a_reason():
    av = available_backends()
    assert set(av) == {"faster-whisper", "mlx"}
    for name, status in av.items():
        assert status == "ready" or len(status) > 10, \
            f"{name} must say why it is unusable, got {status!r}"


def test_pick_backend_refuses_an_unavailable_choice_by_name():
    av = available_backends()
    missing = [n for n, s in av.items() if s != "ready"]
    if not missing:
        return                                   # everything installed here
    try:
        pick_backend(missing[0])
    except AsrUnavailable as e:
        assert missing[0] in str(e)
    else:
        raise AssertionError("should have refused an unavailable backend")


def test_pick_backend_auto_returns_something_usable_or_explains():
    av = available_backends()
    try:
        chosen = pick_backend("auto")
        assert av[chosen] == "ready"
    except AsrUnavailable as e:
        assert "pip install" in str(e)            # must be actionable


def test_transcribe_refuses_an_unavailable_backend_before_downloading():
    """The guard must sit at the point of use, not only on the callers.

    Regression: entry points resolved the backend but transcribe() defaulted to
    faster-whisper, so any other caller reached the old failure after the audio
    had already been fetched.
    """
    import inspect

    from vke.transcripts import asr as asr_mod

    assert inspect.signature(asr_mod.transcribe).parameters["backend"].default == "auto"

    av = available_backends()
    missing = [n for n, s in av.items() if s != "ready"]
    if not missing:
        return

    class Ref:
        video_id, title, url = "x", "t", "https://example.invalid/x"

    downloaded = []
    original = asr_mod._download_audio
    asr_mod._download_audio = lambda *a, **k: downloaded.append(1)
    try:
        asr_mod.transcribe(Ref(), backend=missing[0])
    except AsrUnavailable:
        pass
    else:
        raise AssertionError("should have refused before downloading")
    finally:
        asr_mod._download_audio = original
    assert not downloaded, "refused only after downloading audio"


def test_failure_message_explains_rather_than_just_reporting():
    av = available_backends()
    missing = [n for n, s in av.items() if s != "ready"]
    if not missing:
        return
    try:
        pick_backend(missing[0])
    except AsrUnavailable as e:
        text = str(e)
        for expected in ("platform", "backends", "pip install", "restart"):
            assert expected in text, f"diagnosis is missing {expected!r}"


def test_serve_steps_past_a_busy_port():
    """A stale copy of the tool holding the port must not stop a new one.

    Regression: an older server kept port 7864 and kept serving its own stale
    code, so every fix looked like it had not worked.
    """
    from vke import ui

    tried = []

    class Busy(OSError):
        def __init__(self):
            super().__init__(48, "Address already in use")
            self.errno = 48

    class FakeServer:
        def __init__(self, addr, handler):
            tried.append(addr[1])
            if addr[1] < 7866:                  # first two ports occupied
                raise Busy()

        def serve_forever(self):
            raise KeyboardInterrupt              # stop immediately

        def shutdown(self):
            pass

    real_server, real_holder = ui.ThreadingHTTPServer, ui._port_holder
    ui.ThreadingHTTPServer = FakeServer
    ui._port_holder = lambda port: "fake-holder"
    try:
        ui.serve(port=7864, open_browser=False)
    finally:
        ui.ThreadingHTTPServer, ui._port_holder = real_server, real_holder

    assert tried == [7864, 7865, 7866], f"should step forward, tried {tried}"


def test_serve_explains_when_every_port_is_taken():
    from vke import ui

    class AlwaysBusy:
        def __init__(self, addr, handler):
            e = OSError(48, "Address already in use")
            e.errno = 48
            raise e

    real_server, real_holder = ui.ThreadingHTTPServer, ui._port_holder
    ui.ThreadingHTTPServer = AlwaysBusy
    ui._port_holder = lambda port: "Python 999 excoms"
    try:
        ui.serve(port=7864, open_browser=False, max_tries=3)
    except SystemExit as e:
        text = str(e)
        assert "7864-7866" in text
        assert "lsof -ti:7864" in text          # must be actionable
        assert "Python 999" in text             # must name the holder
    else:
        raise AssertionError("should have exited with an explanation")
    finally:
        ui.ThreadingHTTPServer, ui._port_holder = real_server, real_holder


def test_transcript_char_rate():
    t = Transcript("v", "t", "u", "x" * 800, "ur", "asr:test", 100.0)
    assert t.char_rate == 8.0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  pass  {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {name}: {e}")
    print(f"\n{'all passed' if not failures else str(failures) + ' failed'}")
    sys.exit(1 if failures else 0)
