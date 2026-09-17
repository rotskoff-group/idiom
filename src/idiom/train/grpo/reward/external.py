"""Persistent subprocess rewards using newline-delimited JSON.

Requests contain {"sequences": [...]}; responses contain {"scores": [...]} or
{"error": "..."}. Scores follow input order; shaping is applied in IDiom.
Run python -m idiom.train.grpo.reward.external --help to check a scorer command.
"""

from __future__ import annotations

import argparse
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
import time

from idiom.train.grpo.reward.resolve import SHAPING_ALIASES, Reward, build_from_spec


def _argv(cmd) -> list[str]:
    """Return a command as argv, accepting a list, a JSON list, or a shell-quoted string."""
    if isinstance(cmd, (list, tuple)):
        return [str(a) for a in cmd]
    cmd = cmd.strip()
    if cmd.startswith("["):
        return [str(a) for a in json.loads(cmd)]
    return shlex.split(cmd)


def _label_from_argv(argv: list[str]) -> str:
    """Derive a logging label from the script or executable name."""
    script = next((a for a in argv if a.endswith(".py")), argv[0] if argv else "scorer")
    return os.path.splitext(os.path.basename(script))[0] or "scorer"


def parse_response(line: str, n: int) -> list[float]:
    """Validate one response line and return its scores.

    Args:
        line: One line of the scorer's stdout.
        n: Number of sequences that were sent.

    Returns:
        Exactly n finite scores, in the order the sequences were sent.

    Raises:
        ValueError: If the response is not a JSON object with exactly n finite scores.
        RuntimeError: If the scorer returned an "error" field instead of scores.
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
        raise ValueError(
            f"scorer returned {len(scores)} scores for {n} sequences; scores must "
            f"correspond one-to-one, in the order the sequences were sent"
        )
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


class ScorerProcess:
    """A persistent scorer subprocess using newline-delimited JSON.

    Start lazily, reuse across batches, and retry once if the child exits mid-batch.
    Forward child stderr with a label prefix.

    Attributes:
        argv (list[str]): The scorer command, as arguments.
        cwd (str): Working directory for the child process.
        timeout (float): Deadline in seconds for writing one request and receiving its response.
        env (dict | None): Full environment for the child, or None to inherit this process's.
        label (str): Tag prefixed to the child's forwarded stderr; the script basename by default.
        proc (subprocess.Popen | None): The running child, or None when not started.
    """

    def __init__(
        self,
        cmd,
        *,
        cwd: str | None = None,
        timeout: float = 300.0,
        env: dict | None = None,
        label: str | None = None,
    ) -> None:
        """Record the command and settings without starting the child process.

        Args:
            cmd (str | list[str]): Command that runs the scorer, shell-quoted or an argument list.
            cwd: Working directory for the child; the current directory if None.
            timeout: Deadline in seconds for writing one request and receiving its response.
            env: Environment variables for the child, layered over this process's own environment;
                None passes it through unchanged.
            label: Prefix for child stderr; defaults to the script basename.

        Raises:
            ValueError: If timeout is not positive and finite.
        """
        self.argv = _argv(cmd)
        self.cwd = cwd or os.getcwd()
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.timeout = timeout
        self.env = {**os.environ, **{k: str(v) for k, v in (env or {}).items()}} if env else None
        self.label = label or _label_from_argv(self.argv)
        self.proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Launch the child and begin draining its streams.

        The child runs in its own process group and is terminated at interpreter exit.

        Raises:
            RuntimeError: If the command cannot be run.
        """
        try:
            self.proc = subprocess.Popen(
                self.argv,
                cwd=self.cwd,
                env=self.env,
                text=True,
                bufsize=1,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # so a timeout can kill the whole process group
            )
        except OSError as e:
            raise RuntimeError(f"could not run scorer command {self.argv!r}: {e}") from e

        self._q = queue.Queue()
        self._threads = [
            threading.Thread(target=self._pump_stdout, args=(self.proc, self._q), daemon=True),
            threading.Thread(target=self._pump_stderr, args=(self.proc,), daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        atexit.register(self.stop)

    def _pump_stdout(self, proc: subprocess.Popen, q: queue.Queue) -> None:
        """Forward the child's stdout lines to a queue, putting None at EOF."""
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        """Forward the child's stderr lines to this process's stderr, prefixed with the label."""
        for line in proc.stderr:
            print(f"[{self.label}] {line.rstrip()}", file=sys.stderr, flush=True)

    def stop(self) -> None:
        """Terminate the child's process group if it is still running."""
        proc, self.proc = self.proc, None
        atexit.unregister(self.stop)
        if proc is None:
            return
        # Signal the group even if its leader exited: descendants may still hold pipes or GPUs
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads = []
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()

    def roundtrip(self, seqs: list[str]) -> list[float]:
        """Send one batch to the running child and read one response.

        Args:
            seqs: Sequences to score.

        Returns:
            One score per sequence, in order.

        Raises:
            TimeoutError: If writing the request or receiving its response exceeds the shared
                deadline; the child is stopped.
            BrokenPipeError: If the child's stdin is closed, or it exits without responding.
            ValueError: If the response is malformed.
            RuntimeError: If the scorer reports an error.
        """
        proc = self.proc
        if proc is None:
            raise RuntimeError("scorer is not running")
        deadline = time.monotonic() + self.timeout
        written: queue.Queue = queue.Queue()

        def write_request():
            """Write and flush a JSON request, reporting completion or a pipe error through the queue."""
            try:
                proc.stdin.write(json.dumps({"sequences": seqs}, separators=(",", ":")) + "\n")
                proc.stdin.flush()
                written.put(None)
            except (BrokenPipeError, OSError) as e:
                written.put(e)

        writer = threading.Thread(target=write_request, daemon=True)
        self._threads.append(writer)
        writer.start()
        try:
            error = written.get(timeout=max(0, deadline - time.monotonic()))
            if error is not None:
                raise BrokenPipeError(f"scorer stdin closed: {error}") from error
            line = self._q.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty:
            self.stop()
            raise TimeoutError(
                f"scorer round trip exceeded {self.timeout:g}s for "
                f"{len(seqs)} sequences (raise the term's timeout)"
            ) from None
        finally:
            if writer in self._threads:
                writer.join()
                self._threads.remove(writer)
        if line is None:
            code = self.proc.poll() if self.proc else None
            raise BrokenPipeError(f"scorer exited (returncode={code}) without responding")
        return parse_response(line, len(seqs))

    def score(self, seqs: list[str]) -> list[float]:
        """Score a batch, starting the child if it is not running.

        If the child dies during the batch, it is restarted once and the batch is retried.

        Args:
            seqs: Sequences to score.

        Returns:
            One score per sequence, in order.

        Raises:
            RuntimeError: If the child cannot be started or reports an error.
            BrokenPipeError: If the child dies again after the restart.
            TimeoutError: If a request/response exchange exceeds the deadline.
            ValueError: If the response is malformed or contains invalid scores.
        """
        if self.proc is None or self.proc.poll() is not None:
            self.stop()
            self.start()
        try:
            return self.roundtrip(seqs)
        except BrokenPipeError as e:
            print(f"[{self.label}] restarting scorer after: {e}", file=sys.stderr, flush=True)
            self.stop()
            self.start()
            return self.roundtrip(seqs)


def scorer(
    cmd,
    *,
    timeout: float = 300.0,
    maxlen: int = 0,
    cwd: str | None = None,
    env: dict | None = None,
    cache_max: int = 100_000,
    label: str | None = None,
) -> Reward:
    """Return a reward with its own lazy subprocess and score cache.

    Empty strings score 0; duplicate sequences are sent once. Before storing new scores,
    clear the cache if it already exceeds cache_max. Set cache_max=0 to disable caching.

    Args:
        cmd (str | list[str]): Command that runs the scorer, shell-quoted or an argument list.
        timeout: Deadline in seconds for writing one request and receiving its response.
        maxlen: Truncate sequences to this length before sending; 0 sends them whole.
        cwd: Working directory for the child; the current directory if None.
        env: Environment variables set for the child, over this process's own.
        cache_max: Number of cached sequences above which the cache is cleared; 0 disables caching.
        label: Prefix for child stderr; defaults to the script basename.

    Returns:
        A callable mapping a list of IDRs to raw scores in the same order.

    Raises:
        ValueError: If timeout is not positive and finite, or cache_max is negative.
    """
    child = ScorerProcess(cmd, cwd=cwd, timeout=timeout, env=env, label=label)
    if cache_max < 0:
        raise ValueError("cache_max must be nonnegative")
    cache: dict[str, float] = {}

    def reward(idrs: list[str]) -> list[float]:
        """Score uncached nonempty IDRs through the child process and restore the input order."""
        todo = list(dict.fromkeys(s for s in idrs if s and s not in cache))
        batch = {s: cache[s] for s in idrs if s and s in cache}
        if todo:
            values = child.score([s[:maxlen] if maxlen else s for s in todo])
            if len(cache) > cache_max:
                cache.clear()
            batch.update(zip(todo, values))
            if cache_max:
                cache.update(zip(todo, values))
        return [batch[s] if s else 0.0 for s in idrs]

    return reward


def check(cmd, shaping_spec: dict | None = None, seqs: list[str] | None = None) -> int:
    """Run a scorer command over a few sequences and print its raw and shaped rewards.

    Args:
        cmd (str | list[str]): The scorer command to check.
        shaping_spec: A term's shaping spec -- a name plus that rule's arguments -- or None to show
            the raw reward unshaped.
        seqs: Sequences to score; a small built-in set if None.

    Returns:
        0 if the scorer answered, 1 if it failed.
    """
    seqs = seqs or [
        "MEEEKKKKSSSTTTDDDQQQQNNNN",
        "GSGSGSGSGSGSGSGSGSGSGSGSGSGSGS",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    ]
    shaping = build_from_spec(shaping_spec or "identity", SHAPING_ALIASES, "shaping", "--shaping")
    print(f"command : {cmd}")
    print(f"shaping : {shaping_spec or 'none (the raw reward is used as-is)'}")
    child = ScorerProcess(cmd, timeout=300.0)
    t0 = time.monotonic()
    try:
        values = child.score(seqs)
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        child.stop()
    print(f"\nstartup + {len(seqs)} sequences in {time.monotonic() - t0:.1f}s\n")
    print(f"{'raw':>12}  {'shaped':>8}  sequence")
    for s, v in zip(seqs, values):
        print(f"{v:12.4f}  {shaping(v):8.4f}  {s[:44]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and check an external scorer command.

    Args:
        argv: Argument list; sys.argv[1:] if None.

    Returns:
        Process exit status.
    """
    p = argparse.ArgumentParser(description="Check an external reward scorer command.")
    p.add_argument("--cmd", required=True, help="command that runs the scorer")
    p.add_argument(
        "--shaping",
        default="identity",
        help="shaping rule applied to the raw reward: a shipped name, or module:function",
    )
    p.add_argument("--target", type=float, default=None, help="target for the shaping")
    p.add_argument("--width", type=float, default=None, help="tolerance as a fraction of the target")
    p.add_argument("sequences", nargs="*", help="sequences to score instead of the built-in set")
    args = p.parse_args(argv)

    spec = {"name": args.shaping}
    if args.target is not None:
        spec["target"] = args.target
    if args.width is not None:
        spec["width"] = args.width
    try:
        return check(args.cmd, None if args.shaping == "identity" else spec, args.sequences or None)
    except ValueError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
