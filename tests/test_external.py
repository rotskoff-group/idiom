"""Tests for the external-scorer adapter, driven by fake scorer subprocesses.

No reward model is involved: each test writes a small script that speaks the protocol (or misbehaves
in a specific way) and checks that the adapter handles it. The failure modes matter more than the
happy path -- a scorer that returns the wrong number of scores would silently misalign rewards with
completions.
"""

import sys
import textwrap

import pytest

from idiom.train.grpo.reward.external import (
    Scorer,
    make_external_reward,
    parse_response,
)
from idiom.rewards import rewards_path
from idiom.train.grpo.reward.registry import Batch


def _scorer(tmp_path, body, name="fake_scorer.py", **kw):
    """Write a fake scorer script and return a Scorer that runs it."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return Scorer(f"{sys.executable} {path}", cwd=str(tmp_path), **kw)


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


# ---------------------------------------------------------------- Scorer process


def test_scorer_roundtrip(tmp_path):
    s = _scorer(tmp_path, ECHO_LENGTHS)
    try:
        assert s.score(["AAA", "CCCCC"]) == [3.0, 5.0]
        assert s.score(["MK"]) == [2.0]  # same process reused across batches
    finally:
        s.stop()


def test_scorer_handshake_fails_on_bad_command(tmp_path):
    s = Scorer(f"{sys.executable} {tmp_path / 'does_not_exist.py'}", cwd=str(tmp_path), timeout=30)
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


def test_make_external_reward_batches_dedups_and_caches(tmp_path):
    """Empty strings cost no round trip, duplicates are sent once, repeats hit the cache."""
    counter = tmp_path / "calls.txt"
    path = _batched_scorer_file(tmp_path, counter)
    reward = make_external_reward(f"{sys.executable} {path}", cwd=str(tmp_path))
    assert reward(["AAA", "", "AAA", "CCCCC"], Batch(4)) == [3.0, 0.0, 3.0, 5.0]
    assert reward(["AAA", "GG"], Batch(2)) == [3.0, 2.0]

    batches = [line for line in counter.read_text().splitlines() if line]
    assert batches[0] == '["MKVGSDEQ"]'      # the handshake
    assert batches[1] == '["AAA", "CCCCC"]'  # deduped, empty dropped
    assert batches[2] == '["GG"]'            # only the uncached sequence


def test_make_external_reward_returns_the_raw_value(tmp_path):
    # the scorer reports its own units and stops there; shaping is the term's job, not the
    # subprocess's, so the objective is retuned without touching that environment
    reward = make_external_reward(f"{sys.executable} {_scorer_path(tmp_path)}", cwd=str(tmp_path))
    assert reward(["AAA", "AAAAA"], Batch(2)) == [3.0, 5.0]


def test_two_external_rewards_are_independent(tmp_path):
    # two scorers in one run get their own process and cache, and must not share global state
    m1 = make_external_reward(f"{sys.executable} {_scorer_path(tmp_path)}", cwd=str(tmp_path))
    m2 = make_external_reward(f"{sys.executable} {_scorer_path(tmp_path)}", cwd=str(tmp_path),
                               maxlen=2)
    assert m1(["AAAAA"], Batch(1))[0] == pytest.approx(5.0)
    assert m2(["AAAAA"], Batch(1))[0] == pytest.approx(2.0)  # truncated before it was sent


def test_a_command_can_be_given_as_an_argument_list(tmp_path):
    # the escape hatch from shell quoting: a path with a space in it survives unsplit
    path = _scorer_path(tmp_path, name="len scorer.py")
    reward = make_external_reward([sys.executable, str(path)], cwd=str(tmp_path))
    assert reward(["AAA"], Batch(1)) == [3.0]


def _scorer_path(tmp_path, name="len_scorer.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(ECHO_LENGTHS))
    return path


def test_shipped_sparrow_scorer_speaks_the_protocol(tmp_path, monkeypatch):
    """The shipped scorer must work, since users copy it as their starting template.

    sparrow itself is a heavy build, so a stub package standing in for it is put on the child's
    import path. That leaves the real script's argument parsing, protocol loop, and error handling
    under test without a network round trip -- and the script strips its own directory from
    sys.path precisely so the stub is what "import sparrow" finds.
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
    scorer = Scorer([sys.executable, str(rewards_path("external_rewards/sparrow.py")),
                     "--property", "radius_of_gyration"], timeout=30)
    try:
        assert scorer.score(["FWY", "AAAAA", ""]) == [6.0, 10.0, 0.0]
    finally:
        scorer.stop()
