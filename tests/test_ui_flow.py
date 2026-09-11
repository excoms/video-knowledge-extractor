"""End-to-end exercise of the path the web interface actually runs.

Everything real except the two external boundaries: YouTube and the model.
This exists because unit-testing the pieces missed a NameError that made the
UI fail the instant a user pressed Start — the pieces were all fine, the
wiring was not.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vke import ui                                            # noqa: E402
from vke.sources import VideoRef                              # noqa: E402
from vke.transcripts import Segment, Transcript               # noqa: E402

SAMPLE = ("واجب الوجود ایسے وجود کو کہا جاتا ہے جس کا موجود ہونا ایسا لابدی ہو "
          "کہ اس کے معدوم ہونے کا تصور بھی غلط ہو ممکن الوجود اسے کہا جاتا ہے")


class StubProvider:
    name = "stub"

    def complete(self, system, user, max_tokens=4000):
        if "design compact JSON schemas" in system:
            return ('{"record_name":"argument","fields":['
                    '{"name":"claim","description":"one sentence"},'
                    '{"name":"vulnerability","description":"weak point"}]}')
        if "given every record extracted" in system:
            return ('{"summary":"one recurring problem.","patterns":[],'
                    '"contradictions":[],"strongest":[]}')
        return json.dumps([{
            "fields": {"claim": "The universe is contingent",
                       "vulnerability": "Fallacy of composition"},
            "quote": "واجب الوجود ایسے وجود کو کہا جاتا ہے",
            "speaker": "Host", "confidence": "high"}], ensure_ascii=False)


def _run(tmp: Path, cfg_extra=None, calls=None):
    """Drive run_job with the external world stubbed out.

    `calls` collects one entry per transcription, so a caller can assert that
    finished work was not repeated.
    """
    calls = calls if calls is not None else []
    ref = VideoRef("vid1", "تجرباتی ویڈیو", "https://www.youtube.com/watch?v=vid1")
    segs = [Segment(i * 2.0, i * 2.0 + 2.0, w) for i, w in enumerate(SAMPLE.split())]
    transcript = Transcript("vid1", "تجرباتی ویڈیو", ref.url, SAMPLE, "ur",
                            "asr:stub", 60.0, segs)

    import vke.sources as sources_mod
    import vke.transcripts as tr_mod
    import vke.providers as prov_mod

    saved = (sources_mod.resolve_many, tr_mod.transcribe, prov_mod.get_provider,
             prov_mod.autodetect)
    sources_mod.resolve_many = lambda urls, limits=None: [ref]

    def _transcribe(*a, **k):
        calls.append(1)
        return transcript

    tr_mod.transcribe = _transcribe
    prov_mod.get_provider = lambda name, model=None: StubProvider()
    prov_mod.autodetect = lambda: "stub"

    ui.JOB.update(state="idle", message="", videos=[], records=[], error="")
    cfg = {"urls": ref.url, "ask": "Find every argument and its weakness",
           "shape": "One section each", "outdir": str(tmp),
           "cache": str(tmp / "_cache"),      # never touch the user's real cache
           "asr": "auto", "provider": "stub", "output_language": "English"}
    cfg.update(cfg_extra or {})
    try:
        ui.run_job(cfg)
    finally:
        (sources_mod.resolve_many, tr_mod.transcribe,
         prov_mod.get_provider, prov_mod.autodetect) = saved
    return dict(ui.JOB)


def test_full_run_reaches_done_and_writes_every_output(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    job = _run(tmp)

    assert job["error"] == "", f"run reported an error: {job['error']}"
    assert job["state"] == "done", f"expected done, got {job['state']!r}"
    assert job["records"], "no records produced"

    r = job["records"][0]
    for key in ("video_id", "quote_original", "timestamp", "record", "confidence"):
        assert key in r, f"record missing {key}"
    assert set(r["record"]) == {"claim", "vulnerability"}

    for name in ("analysis.json", "analysis.md"):
        f = tmp / name
        assert f.exists() and f.stat().st_size > 0, f"{name} not written"
    assert (tmp / "transcripts" / "vid1.txt").exists(), "transcript not saved"

    data = json.loads((tmp / "analysis.json").read_text())
    assert data["meta"]["record_count"] == len(job["records"])


def test_transcribe_only_stops_before_analysis(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    job = _run(tmp, {"transcribe_only": True})
    assert job["state"] == "done" and job["error"] == ""
    assert not job["records"], "should not analyse when transcribe_only is set"
    assert (tmp / "transcripts" / "vid1.txt").exists()
    assert not (tmp / "analysis.md").exists()


def test_a_second_run_does_not_transcribe_again(tmp_path=None):
    """The checkpoint must actually prevent the expensive work being repeated.

    Asserting on a status label is not enough — the label is overwritten by the
    analysis phase. Count the transcriptions instead.
    """
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())

    first = []
    _run(tmp, calls=first)
    assert len(first) == 1, f"first run should transcribe once, did {len(first)}"

    second = []
    job = _run(tmp, calls=second)
    assert job["state"] == "done" and job["error"] == ""
    assert second == [], "second run re-transcribed work already checkpointed"
    assert job["records"], "second run should still produce analysis"


def test_bad_input_is_reported_not_raised(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    job = _run(tmp, {"urls": "   "})
    assert job["state"] == "error"
    assert "link" in job["error"].lower()


def test_a_run_folder_is_self_contained_even_when_reusing_the_cache():
    """The point of a folder per run is that you can move or send it.

    Regression: the CLI skipped cached videos entirely, leaving the run folder
    empty — the transcript existed only in the shared cache.
    """
    import tempfile

    from vke.store import Store
    from vke.transcripts import Segment

    tmp = Path(tempfile.mkdtemp())
    cache, run_a, run_b = tmp / "cache", tmp / "run-a", tmp / "run-b"

    first = Store(run_a, cache=cache)
    first.put_transcript("vidX", "TITLE: T\nURL: u\nDURATION: 60s\nLANGUAGE: ur",
                         "some words here",
                         [Segment(0.0, 2.0, "some"), Segment(2.0, 4.0, "words")])

    second = Store(run_b, cache=cache)
    assert second.done("vidX"), "cache should satisfy a fresh run"
    assert second.adopt_from_cache("vidX")
    assert (run_b / "transcripts" / "vidX.txt").exists(), "run folder must hold it"
    assert (run_b / "transcripts" / "vidX.segments.json").exists(), "timings too"
    assert second.state["vidX"]["status"] == "ok", "state must be recorded"
    assert second.get_segments("vidX"), "segments must load in the new run"


def test_every_endpoint_the_page_calls_is_handled():
    """The page and the server must not drift apart.

    A route the JS fetches but the server does not handle fails silently — the
    .catch() swallows it and a control simply never populates.
    """
    import re

    page = (Path(ui.__file__).parent / "static" / "index.html").read_text()
    wanted = set(re.findall(r"fetch\(\s*['\"](/api/[a-z_]+)['\"]", page))
    assert wanted, "no fetch calls found — did the page change shape?"

    handler = Path(ui.__file__).read_text()
    for route in sorted(wanted):
        assert f'"{route}"' in handler, f"page calls {route} but the server never handles it"


def test_the_page_is_served_uncacheable():
    """A cached page keeps submitting stale choices after the server is fixed."""
    handler = Path(ui.__file__).read_text()
    assert "no-store" in handler, "responses must not be cacheable"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  pass  {name}")
            except Exception as e:
                failures += 1
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{'all passed' if not failures else str(failures) + ' failed'}")
    sys.exit(1 if failures else 0)
