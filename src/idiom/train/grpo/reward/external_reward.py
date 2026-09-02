"""Run an external reward model, in its own environment, as a GRPO reward term.

A reward model that cannot be installed next to IDiom (a different python, torch, or CUDA) runs as a
subprocess "scorer" spoken to over newline-delimited JSON, one exchange per GRPO step:

    ->  {"sequences": ["ACDEF...", "GHIKL..."]}
    <-  {"scores": [24.8, 31.2]}          # or {"error": "..."}

A scorer imports nothing from IDiom, so it can live in any venv, conda env, or container. Each
external term in configs/grpo.yaml carries its own cmd, so several coexist in one run; the scorer
returns a raw value and the term's target/width band it to a reward (see the reward section of the
top-level README).

Check a command before spending a GPU allocation:

    uv run python -m idiom.train.grpo.reward.external_reward --cmd "<scorer command>" --target 25 --width 3
"""

from __future__ import annotations

import atexit
import json
import math
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading

_PROBE = "MKVGSDEQ"  # handshake sequence: a valid IDR every scorer should be able to score


def _argv(cmd: str) -> list[str]:
    """Split a command string into argv, accepting a JSON list for arguments shlex would mangle."""
    cmd = cmd.strip()
    if cmd.startswith("["):
        return [str(a) for a in json.loads(cmd)]
    return shlex.split(cmd)


def parse_response(line: str, n: int) -> list[float]:
    """Validate one NDJSON response line and return its n scores.

    The length check is the important one: a scorer that returns the wrong number of scores would
    silently misalign rewards with completions, which corrupts training without raising anywhere.

    Args:
        line (str): One line of the scorer's stdout.
        n (int): Number of sequences that were sent.

    Returns:
        list[float]: The finite scores, in the order the sequences were sent.

    Raises:
        ValueError: If the line is not a JSON object, has no scores, has the wrong number of
            scores, or holds a value that is not a finite number.
        RuntimeError: If the scorer reported an error instead of scores.
    """
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"scorer wrote a non-JSON line: {line[:200]!r}") from e
    if not isinstance(msg, dict):
        raise ValueError(f"scorer response must be a JSON object, got {type(msg).__name__}")
    if "error" in msg:
        raise RuntimeError(f"scorer reported an error: {msg['error']}")
    scores = msg.get("scores")
    if not isinstance(scores, list):
        raise ValueError(f"scorer response has no 'scores' list (keys: {sorted(msg)})")
    if len(scores) != n:
        raise ValueError(f"scorer returned {len(scores)} scores for {n} sequences; scores must "
                         f"correspond one-to-one, in the order the sequences were sent")
    out = []
    for i, s in enumerate(scores):
        try:
            v = float(s)
        except (TypeError, ValueError):
            raise ValueError(f"score {i} is not a number: {s!r}") from None
        if not math.isfinite(v):
            raise ValueError(f"score {i} is not finite: {v!r}")
        out.append(v)
    return out


class Scorer:
    """A persistent scorer subprocess spoken to in newline-delimited JSON.

    The child is started once and reused, so the reward model loads once rather than per step. Its
    stderr is forwarded to ours (debugging a foreign environment blind is hopeless), a handshake
    runs at startup so a broken command fails immediately, and a child that dies mid-run is
    restarted once before the error is allowed to stop training.
    """

    def __init__(self, cmd: str, *, cwd: str | None = None, timeout: float = 300.0,
                 label: str = "external") -> None:
        """Initialize the scorer without starting the child process.

        Args:
            cmd (str): Command that runs the scorer, shell-quoted or a JSON argv list.
            cwd (str | None): Working directory for the child; defaults to the current directory.
            timeout (float): Seconds to wait for a single response.
            label (str): Short tag used to prefix the child's forwarded stderr.
        """
        self.argv = _argv(cmd)
        self.cwd = cwd or os.getcwd()
        self.timeout = timeout
        self.label = label
        self.proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()

    def start(self) -> None:
        """Launch the child, begin draining its streams, and run the startup handshake.

        Raises:
            RuntimeError: If the command cannot be run or fails to answer the handshake.
        """
        try:
            self.proc = subprocess.Popen(
                self.argv, cwd=self.cwd, text=True, bufsize=1,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,  # so a timeout can kill the whole process group
            )
        except OSError as e:
            raise RuntimeError(f"could not run scorer command {self.argv!r}: {e}") from e

        self._q = queue.Queue()
        threading.Thread(target=self._pump_stdout, args=(self.proc, self._q), daemon=True).start()
        threading.Thread(target=self._pump_stderr, args=(self.proc,), daemon=True).start()
        atexit.register(self.stop)

        try:
            self.roundtrip([_PROBE])
        except Exception as e:
            self.stop()
            raise RuntimeError(
                f"scorer handshake failed: {' '.join(self.argv)}\n  {e}\n"
                f"  the scorer must read one JSON line from stdin and write "
                f"{{\"scores\": [...]}} to stdout; see the reward section of the top-level README.md") from e

    def _pump_stdout(self, proc: subprocess.Popen, q: queue.Queue) -> None:
        """Forward the child's stdout lines to a queue, ending with None at EOF."""
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        """Forward the child's stderr to ours, prefixed, so its tracebacks reach the job log."""
        for line in proc.stderr:
            print(f"[{self.label}] {line.rstrip()}", file=sys.stderr, flush=True)

    def stop(self) -> None:
        """Terminate the child process group, if it is still running."""
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.terminate()

    def roundtrip(self, seqs: list[str]) -> list[float]:
        """Send one batch and read one response.

        Args:
            seqs (list[str]): Sequences to score.

        Returns:
            list[float]: One score per sequence, in order.

        Raises:
            TimeoutError: If no response arrives within the timeout.
            BrokenPipeError: If the child exits without responding.
        """
        try:
            self.proc.stdin.write(json.dumps({"sequences": seqs}, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise BrokenPipeError(f"scorer stdin closed: {e}") from e
        try:
            line = self._q.get(timeout=self.timeout)
        except queue.Empty:
            self.stop()
            raise TimeoutError(f"scorer sent no response within {self.timeout:g}s for "
                               f"{len(seqs)} sequences (raise the term's timeout)") from None
        if line is None:
            code = self.proc.poll() if self.proc else None
            raise BrokenPipeError(f"scorer exited (returncode={code}) without responding")
        return parse_response(line, len(seqs))

    def score(self, seqs: list[str]) -> list[float]:
        """Score a batch, starting or restarting the child as needed.

        Args:
            seqs (list[str]): Sequences to score.

        Returns:
            list[float]: One score per sequence, in order.
        """
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        try:
            return self.roundtrip(seqs)
        except BrokenPipeError as e:
            # the child died mid-run (OOM, segfault); one restart, then let the error stand
            print(f"[{self.label}] restarting scorer after: {e}", file=sys.stderr, flush=True)
            self.stop()
            self.start()
            return self.roundtrip(seqs)


def band(value: float, target: float | None, width: float) -> float:
    """Map a raw scorer value to a reward through a target band, or pass it through.

    Args:
        value (float): The scorer's raw value.
        target (float | None): Band centre; None passes the raw value through unchanged.
        width (float): Band width in the value's own units.

    Returns:
        float: exp(-((value - target) / width)^2 / 2) in (0, 1] when target is set, else value.
    """
    if target is None:
        return value
    d = (value - target) / width
    return math.exp(-0.5 * d * d)


def make_external_reward(cmd: str, *, target: float | None = None, width: float = 1.0,
                         timeout: float = 300.0, maxlen: int = 0, cwd: str | None = None,
                         cache_max: int = 100_000, label: str = "external"):
    """Build a batched reward that scores a whole GRPO step through one external scorer subprocess.

    Each call makes an independent scorer with its own process and cache, so several external terms
    (each its own command, environment and target) coexist in one run. Empty IDRs score 0.0 without
    a round trip, and duplicate sequences within a step are sent once (GRPO produces both
    constantly).

    Args:
        cmd (str): Command that runs the scorer, shell-quoted or a JSON argv list.
        target (float | None): Band centre; None uses the raw scorer value as the reward.
        width (float): Band width in the value's own units.
        timeout (float): Seconds to wait for one response.
        maxlen (int): Truncate sequences to this length before sending (0 = no truncation).
        cwd (str | None): Working directory for the child; defaults to the current directory.
        cache_max (int): Maximum cached sequences before the cache is cleared.
        label (str): Short tag for the term, used to prefix the child's stderr.

    Returns:
        Callable[[list[str], int], list[float]]: Maps (idrs, group_size) to one reward per IDR.
    """
    scorer = Scorer(cmd, cwd=cwd, timeout=timeout, label=label)
    cache: dict[str, float] = {}

    def reward(idrs: list[str], group_size: int) -> list[float]:
        todo = list(dict.fromkeys(s for s in idrs if s and s not in cache))
        if todo:
            values = scorer.score([s[:maxlen] if maxlen else s for s in todo])
            if len(cache) > cache_max:
                cache.clear()
            cache.update(zip(todo, values))
        return [band(cache[s], target, width) if s else 0.0 for s in idrs]

    return reward


def check(cmd: str, target: float | None, width: float, seqs: list[str] | None = None) -> int:
    """Run a command over a few sequences and print the raw values and band rewards it produces.

    Args:
        cmd (str): The scorer command to check.
        target (float | None): Band centre, or None to show the raw value as the reward.
        width (float): Band width.
        seqs (list[str] | None): Sequences to score; a small built-in set when None.

    Returns:
        int: 0 if the scorer answered, 1 if it failed.
    """
    import time

    seqs = seqs or ["MEEEKKKKSSSTTTDDDQQQQNNNN",
                    "GSGSGSGSGSGSGSGSGSGSGSGSGSGSGS",
                    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"]
    print(f"command : {cmd or '(none -- pass --cmd)'}")
    print(f"reward  : {'raw value' if target is None else f'band(target={target:g}, width={width:g})'}")
    scorer = Scorer(cmd, timeout=300.0)
    t0 = time.monotonic()
    try:
        values = scorer.score(seqs)
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        scorer.stop()
    print(f"\nstartup + {len(seqs)} sequences in {time.monotonic() - t0:.1f}s\n")
    print(f"{'raw':>12}  {'reward':>8}  sequence")
    for s, v in zip(seqs, values):
        print(f"{v:12.4f}  {band(v, target, width):8.4f}  {s[:44]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for checking an external scorer command.

    Args:
        argv (list[str] | None): Argument list; sys.argv[1:] when None.

    Returns:
        int: Process exit status.
    """
    import argparse

    p = argparse.ArgumentParser(description="Check an external reward scorer command.")
    p.add_argument("--cmd", required=True, help="command that runs the scorer")
    p.add_argument("--target", type=float, default=None, help="band centre (default: raw value)")
    p.add_argument("--width", type=float, default=1.0, help="band width")
    p.add_argument("sequences", nargs="*", help="sequences to score instead of the built-in set")
    args = p.parse_args(argv)
    return check(args.cmd, args.target, args.width, args.sequences or None)


if __name__ == "__main__":
    sys.exit(main())
