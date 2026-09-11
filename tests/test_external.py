"""Tests for the external-scorer adapter, driven by fake scorer subprocesses.

No reward model is involved: each test writes a small script that speaks the protocol, or misbehaves
in a specific way, and checks that the adapter handles it.
"""

import sys
import textwrap
from pathlib import Path

import pytest

from idiom.train.grpo.reward.external import (
    ScorerProcess,
    _label_from_argv,
    parse_response,
    scorer,
)

REPO = Path(__file__).resolve().parents[1]


def _scorer(tmp_path, body, name="fake_scorer.py", **kw):
    """Write a fake scorer script and return a ScorerProcess that runs it."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return ScorerProcess(f"{sys.executable} {path}", cwd=str(tmp_path), **kw)


ECHO_LENGTHS = """
    import json, sys
    for line in sys.stdin:
        if not line.strip():
            continue
        seqs = json.loads(line)["sequences"]
        print(json.dumps({"scores": [float(len(s)) for s in seqs]}), flush=True)
"""




def test_parse_response_happy():
    assert parse_response('{"scores": [1, 2.5]}', 2) == [1.0, 2.5]


def test_parse_response_rejects_length_mismatch():
    with pytest.raises(ValueError, match="2 scores for 3 sequences"):
        parse_response('{"scores": [1, 2]}', 3)


def test_parse_response_rejects_non_finite():
    with pytest.raises(ValueError, match="not finite"):
        parse_response('{"scores": [1, NaN]}', 2)


def test_parse_response_rejects_non_numeric():
    with pytest.raises(ValueError, match="not a number"):
        parse_response('{"scores": ["a", 2]}', 2)


def test_parse_response_surfaces_scorer_error():
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        parse_response('{"error": "CUDA out of memory"}', 2)


def test_parse_response_rejects_garbage():
    with pytest.raises(ValueError, match="non-JSON"):
        parse_response("Traceback (most recent call last):", 1)


def test_parse_response_rejects_missing_scores():
    with pytest.raises(ValueError, match="no 'scores' list"):
        parse_response('{"result": [1]}', 1)




def test_label_defaults_to_the_script_basename(tmp_path):
    assert _label_from_argv(["uv", "run", "--script", "a/b/finches.py", "--mode", "x"]) == "finches"
    assert _label_from_argv(["/usr/bin/python", "/tmp/foo.py"]) == "foo"
    assert _label_from_argv(["mycmd", "--flag"]) == "mycmd"
    assert ScorerProcess("uv run --script x/protgps.py --compartment nucleolus").label == "protgps"




def test_scorer_roundtrip(tmp_path):
    s = _scorer(tmp_path, ECHO_LENGTHS)
    try:
        assert s.score(["AAA", "CCCCC"]) == [3.0, 5.0]
        assert s.score(["MK"]) == [2.0]
    finally:
        s.stop()


def test_scorer_fails_on_bad_command(tmp_path):
    s = ScorerProcess(f"{sys.executable} {tmp_path / 'does_not_exist.py'}", cwd=str(tmp_path), timeout=30)
    with pytest.raises(BrokenPipeError, match="exited"):
        s.score(["AAA"])


def test_scorer_rejects_garbage(tmp_path):
    s = _scorer(tmp_path, """
        import sys
        for line in sys.stdin:
            print("not json at all", flush=True)
    """)
    with pytest.raises(ValueError, match="non-JSON"):
        s.score(["AAA"])
    s.stop()


def test_scorer_restarts_after_the_child_dies(tmp_path):
    # the child exits on the batch containing "DIE"; the adapter must restart and re-serve
    s = _scorer(tmp_path, """
        import json, sys
        for line in sys.stdin:
            if not line.strip():
                continue
            seqs = json.loads(line)["sequences"]
            if "DIE" in seqs:
                sys.exit(1)
            print(json.dumps({"scores": [float(len(x)) for x in seqs]}), flush=True)
    """)
    try:
        assert s.score(["AAA"]) == [3.0]
        with pytest.raises(BrokenPipeError):
            s.score(["DIE"]) # restart happens, then the same batch kills it again
        assert s.score(["AAAA"]) == [4.0]
    finally:
        s.stop()


def test_scorer_times_out_instead_of_hanging(tmp_path):
    s = _scorer(tmp_path, """
        import sys, time
        for line in sys.stdin:
            time.sleep(60)
    """, timeout=1.0)
    with pytest.raises(TimeoutError, match="round trip exceeded"):
        s.score(["AAA"])
    assert s.proc is None


def test_scorer_error_response_propagates(tmp_path):
    s = _scorer(tmp_path, """
        import json, sys
        for line in sys.stdin:
            if not line.strip():
                continue
            seqs = json.loads(line)["sequences"]
            if "BAD" in seqs:
                print(json.dumps({"error": "unsupported residue"}), flush=True)
            else:
                print(json.dumps({"scores": [0.0] * len(seqs)}), flush=True)
    """)
    try:
        with pytest.raises(RuntimeError, match="unsupported residue"):
            s.score(["BAD"])
        assert s.score(["AAA"]) == [0.0]
    finally:
        s.stop()




def _batched_scorer_file(tmp_path, counter):
    """A scorer that returns len(seq) and appends each batch it receives to a counter file."""
    path = tmp_path / "count_scorer.py"
    path.write_text(textwrap.dedent(f"""
        import json, sys
        for line in sys.stdin:
            if not line.strip():
                continue
            seqs = json.loads(line)["sequences"]
            with open({str(counter)!r}, "a") as fh:
                fh.write(json.dumps(seqs) + chr(10))
            print(json.dumps({{"scores": [float(len(x)) for x in seqs]}}), flush=True)
    """))
    return path


def test_scorer_batches_dedups_and_caches(tmp_path):
    counter = tmp_path / "calls.txt"
    path = _batched_scorer_file(tmp_path, counter)
    reward = scorer(f"{sys.executable} {path}", cwd=str(tmp_path))
    assert reward(["AAA", "", "AAA", "CCCCC"]) == [3.0, 0.0, 3.0, 5.0]
    assert reward(["AAA", "GG"]) == [3.0, 2.0]

    batches = [line for line in counter.read_text().splitlines() if line]
    assert batches[0] == '["AAA", "CCCCC"]' # deduped, empty dropped
    assert batches[1] == '["GG"]' # only the uncached sequence


def test_scorer_returns_the_raw_value(tmp_path):
    reward = scorer(f"{sys.executable} {_scorer_path(tmp_path)}", cwd=str(tmp_path))
    assert reward(["AAA", "AAAAA"]) == [3.0, 5.0]


def test_external_scorer_config_applies_shaping_and_weight(tmp_path):
    from idiom.train.grpo.reward import build_reward

    reward = build_reward({"terms": [{
        "label": "external_length",
        "reward": {
            "name": "external_scorer",
            "cmd": [sys.executable, str(_scorer_path(tmp_path))],
        },
        "shaping": {"name": "quadratic", "target": 4, "width": 0.25},
        "weight": 2.0,
    }]})
    totals, breakdown = reward(["AAA", "AAAA", "AAAAA"], 1)
    assert totals == [-2.0, 0.0, -2.0]
    assert [row["external_length_raw"] for row in breakdown] == [3.0, 4.0, 5.0]


def test_two_scorers_are_independent(tmp_path):
    m1 = scorer(f"{sys.executable} {_scorer_path(tmp_path)}", cwd=str(tmp_path))
    m2 = scorer(f"{sys.executable} {_scorer_path(tmp_path)}", cwd=str(tmp_path),
                               maxlen=2)
    assert m1(["AAAAA"])[0] == pytest.approx(5.0)
    assert m2(["AAAAA"])[0] == pytest.approx(2.0) # truncated before it was sent


def test_a_command_can_be_given_as_an_argument_list(tmp_path):
    path = _scorer_path(tmp_path, name="len scorer.py")
    reward = scorer([sys.executable, str(path)], cwd=str(tmp_path))
    assert reward(["AAA"]) == [3.0]


def _scorer_path(tmp_path, name="len_scorer.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(ECHO_LENGTHS))
    return path


def test_shipped_sparrow_scorer_speaks_the_protocol(tmp_path, monkeypatch):
    """Test the real scorer protocol with a stub sparrow package to avoid heavy dependencies."""
    (tmp_path / "sparrow.py").write_text(textwrap.dedent("""
        class _Predictor:
            def __init__(self, seq):
                self.seq = seq
            def radius_of_gyration(self):
                return 2.0 * len(self.seq)

        class Protein:
            def __init__(self, seq):
                self.seq = seq
                self.predictor = _Predictor(seq)
            @property
            def FCR(self):
                return 0.25
    """))
    # Disable site-packages to test the protocol without IDiom or training dependencies
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    scorer = ScorerProcess([sys.executable, "-S", str(REPO / "cookbook/rewards/scorers/sparrow.py"),
                     "--property", "radius_of_gyration"], timeout=30)
    try:
        assert scorer.score(["FWY", "AAAAA", ""]) == [6.0, 10.0, 0.0]
    finally:
        scorer.stop()




SERVE = (
    f"import sys\nsys.path.insert(0, {str(REPO / 'cookbook/rewards/scorers')!r})\n"
    "from _scorer_protocol import serve\nserve(build)\n"
)


def test_serve_scores_a_batch_and_zeros_empties(tmp_path):
    (tmp_path / "s.py").write_text(
        "import json, os, sys\n"
        "def build():\n"
        "    return lambda seqs: [len(s) for s in seqs]\n"
        + SERVE
    )
    sc = ScorerProcess(f"{sys.executable} {tmp_path / 's.py'}", cwd=str(tmp_path), timeout=30)
    try:
        assert sc.score(["FWY", "", "AAAAA"]) == [3.0, 0.0, 5.0]
    finally:
        sc.stop()


def test_serve_turns_a_scorer_exception_into_an_error_response(tmp_path):
    (tmp_path / "s.py").write_text(
        "import json, os, sys\n"
        "def build():\n"
        "    def score_batch(seqs):\n"
        "        raise ValueError('bad seq')\n"
        "    return score_batch\n"
        + SERVE
    )
    sc = ScorerProcess(f"{sys.executable} {tmp_path / 's.py'}", cwd=str(tmp_path), timeout=30)
    with pytest.raises(RuntimeError, match="bad seq"):
        sc.score(["ACDE"])
    sc.stop()


def test_cache_eviction_preserves_current_batch(tmp_path):
    reward = scorer([sys.executable, str(_scorer_path(tmp_path))], cache_max=1)
    assert reward(["AAA", "CCCCC"]) == [3.0, 5.0]
    assert reward(["AAA", "GG"]) == [3.0, 2.0]


def test_cache_can_be_disabled(tmp_path):
    counter = tmp_path / "calls.txt"
    reward = scorer([sys.executable, str(_batched_scorer_file(tmp_path, counter))], cache_max=0)
    assert reward(["AAA", "AAA"]) == [3.0, 3.0]
    assert reward(["AAA"]) == [3.0]
    assert counter.read_text().splitlines() == ['["AAA"]', '["AAA"]']


@pytest.mark.parametrize("values", ["[7]", "[7, 8, 9]", "[float('nan'), 8]", "[float('inf'), 8]"])
def test_serve_rejects_invalid_model_output(tmp_path, values):
    path = tmp_path / "invalid.py"
    path.write_text(f"def build(): return lambda seqs: {values}\n" + SERVE)
    sc = ScorerProcess([sys.executable, str(path)], timeout=5)
    try:
        with pytest.raises(RuntimeError, match="scorer reported an error"):
            sc.score(["AAA", "CCC"])
    finally:
        sc.stop()


def test_serve_validates_requests_and_preserves_protocol(tmp_path):
    import json
    import subprocess

    path = tmp_path / "protocol.py"
    path.write_text(
        "def build():\n"
        "    print('loading model')\n"
        "    def score(seqs):\n"
        "        print('scoring batch')\n"
        "        return [len(s) for s in seqs]\n"
        "    return score\n" + SERVE
    )
    result = subprocess.run(
        [sys.executable, str(path)],
        input='{"sequences":"AAA"}\n{"sequences":[1]}\n{"sequences":["AAA",""]}\n',
        capture_output=True, text=True, check=True, timeout=5,
    )
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert "error" in responses[0] and "error" in responses[1]
    assert responses[2] == {"scores": [3.0, 0.0]}
    assert "loading model" in result.stderr and "scoring batch" in result.stderr


def test_timeout_covers_blocked_request_write(tmp_path):
    sc = _scorer(tmp_path, "import time\ntime.sleep(60)", timeout=0.2)
    try:
        with pytest.raises(TimeoutError):
            sc.score(["A" * 1_000_000])
        assert sc.proc is None
    finally:
        sc.stop()


def test_stop_kills_and_reaps_uncooperative_child(tmp_path):
    sc = _scorer(tmp_path, """
        import json, signal, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for line in sys.stdin:
            seqs = json.loads(line)["sequences"]
            print(json.dumps({"scores": [1] * len(seqs)}), flush=True)
    """)
    try:
        assert sc.score(["AAA"]) == [1.0]
        proc = sc.proc
        sc.stop()
        assert proc.returncode is not None
        assert all(stream.closed for stream in (proc.stdin, proc.stdout, proc.stderr))
    finally:
        sc.stop()
