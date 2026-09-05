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
    parse_response,
    scorer,
)

REPO = Path(__file__).resolve().parents[1]  # the scorers are repository material


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


# ---------------------------------------------------------------- parse_response


def test_parse_response_happy():
    assert parse_response('{"scores": [1, 2.5]}', 2) == [1.0, 2.5]


def test_parse_response_rejects_length_mismatch():
    # the critical check: fewer scores than sequences must never be silently zipped
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


# ---------------------------------------------------------------- ScorerProcess


def test_scorer_roundtrip(tmp_path):
    s = _scorer(tmp_path, ECHO_LENGTHS)
    try:
        assert s.score(["AAA", "CCCCC"]) == [3.0, 5.0]
        assert s.score(["MK"]) == [2.0]  # same process reused across batches
    finally:
        s.stop()


def test_scorer_handshake_fails_on_bad_command(tmp_path):
    s = ScorerProcess(f"{sys.executable} {tmp_path / 'does_not_exist.py'}", cwd=str(tmp_path), timeout=30)
    with pytest.raises(RuntimeError, match="handshake failed"):
        s.score(["AAA"])


def test_scorer_handshake_fails_when_scorer_writes_garbage(tmp_path):
    s = _scorer(tmp_path, """
        import sys
        for line in sys.stdin:
            print("not json at all", flush=True)
    """)
    with pytest.raises(RuntimeError, match="handshake failed"):
        s.score(["AAA"])


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
            s.score(["DIE"])  # restart happens, then the same batch kills it again
        assert s.score(["AAAA"]) == [4.0]  # a healthy batch works again afterwards
    finally:
        s.stop()


def test_scorer_times_out_instead_of_hanging(tmp_path):
    s = _scorer(tmp_path, """
        import sys, time
        for line in sys.stdin:
            time.sleep(60)
    """, timeout=1.0)
    with pytest.raises(RuntimeError, match="handshake failed"):
        s.score(["AAA"])
    assert s.proc is None  # the process group was killed, not left running


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
        assert s.score(["AAA"]) == [0.0]  # in-band errors leave the process alive
    finally:
        s.stop()


# ---------------------------------------------------------------- the batched reward


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
    """Empty strings cost no round trip, duplicates are sent once, repeats hit the cache."""
    counter = tmp_path / "calls.txt"
    path = _batched_scorer_file(tmp_path, counter)
    reward = scorer(f"{sys.executable} {path}", cwd=str(tmp_path))
    assert reward(["AAA", "", "AAA", "CCCCC"]) == [3.0, 0.0, 3.0, 5.0]
    assert reward(["AAA", "GG"]) == [3.0, 2.0]

    batches = [line for line in counter.read_text().splitlines() if line]
    assert batches[0] == '["MKVGSDEQ"]'      # the handshake
    assert batches[1] == '["AAA", "CCCCC"]'  # deduped, empty dropped
    assert batches[2] == '["GG"]'            # only the uncached sequence


def test_scorer_returns_the_raw_value(tmp_path):
    # the scorer reports its own units and stops there; shaping is the term's job, not the
    # subprocess's, so the objective is retuned without touching that environment
    reward = scorer(f"{sys.executable} {_scorer_path(tmp_path)}", cwd=str(tmp_path))
    assert reward(["AAA", "AAAAA"]) == [3.0, 5.0]


def test_two_scorers_are_independent(tmp_path):
    # two scorers in one run get their own process and cache, and must not share global state
    m1 = scorer(f"{sys.executable} {_scorer_path(tmp_path)}", cwd=str(tmp_path))
    m2 = scorer(f"{sys.executable} {_scorer_path(tmp_path)}", cwd=str(tmp_path),
                               maxlen=2)
    assert m1(["AAAAA"])[0] == pytest.approx(5.0)
    assert m2(["AAAAA"])[0] == pytest.approx(2.0)  # truncated before it was sent


def test_a_command_can_be_given_as_an_argument_list(tmp_path):
    # the escape hatch from shell quoting: a path with a space in it survives unsplit
    path = _scorer_path(tmp_path, name="len scorer.py")
    reward = scorer([sys.executable, str(path)], cwd=str(tmp_path))
    assert reward(["AAA"]) == [3.0]


def _scorer_path(tmp_path, name="len_scorer.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(ECHO_LENGTHS))
    return path


def test_shipped_sparrow_scorer_speaks_the_protocol(tmp_path, monkeypatch):
    """The shipped sparrow scorer speaks the protocol against a stub sparrow package.

    sparrow itself is a heavy build, so a stub standing in for it is put on the child's import
    path, leaving the real script's argument parsing, protocol loop, and error handling under test.
    """
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
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # inherited by the scorer subprocess
    scorer = ScorerProcess([sys.executable, str(REPO / "cookbook/rewards/scorers/sparrow.py"),
                     "--property", "radius_of_gyration"], timeout=30)
    try:
        assert scorer.score(["FWY", "AAAAA", ""]) == [6.0, 10.0, 0.0]
    finally:
        scorer.stop()


# ---------------------------------------------------------------- the pasted serve() block


SERVE = (REPO / "cookbook/rewards/scorers/finches.py").read_text()
SERVE = SERVE[SERVE.index("def serve(build):"):]  # the block every scorer pastes verbatim


def test_serve_scores_a_batch_and_zeros_empties(tmp_path):
    """The shared serve() block: build() returns score_batch, empties score 0.0, order preserved."""
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
    """A raise inside score_batch becomes an {"error": ...} the parent surfaces, not a crash."""
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
