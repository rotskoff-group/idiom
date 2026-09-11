# /// script
# requires-python = ">=3.10"
# dependencies = ["biopython>=1.83", "numpy<2"]
# ///
"""Template for an external scorer with its own dependencies; copy and adapt.

This example uses Biopython to calculate isoelectric point (the default) or
molecular weight, selected by --property. It returns raw measurements; IDiom
applies shaping and weights in the training process.

Creating your own scorer
-----------------------
1. Copy this file and _scorer_protocol.py into the same directory. Rename this
   file, for example to my_scorer.py, and keep the helper's filename unchanged.
2. Edit requires-python and dependencies in the script header above. Running
   `uv run --script /path/to/my_scorer.py` creates an environment with those
   dependencies. Install uv first with `python -m pip install uv` if needed.
   The scorer does not need IDiom installed in its environment.
3. Adapt build(): parse any command-line options, validate them, import your
   scoring dependencies, and load model weights or other expensive resources.
   Keep setup here so it happens once per scorer process, not once per batch.
4. Replace score_batch() with your calculation. Return one finite numeric value
   per input sequence, in the same order. A list of Python floats is simplest;
   convert model tensors to CPU values before returning them. Batch predictions
   when your model supports it. Do not reorder or omit sequences.
5. Keep `serve(build)` at the bottom. build() must return the scoring function,
   not a list of scores. The helper handles communication with IDiom.
6. Set reward.name to external_scorer and reward.cmd to your new command in the
   matching YAML file in cookbook/scripts/training/grpo/. Adjust the term's
   label, shaping, target, and weight for your measurement's units and scale.

Input and output contract
-------------------------
score_batch() receives a list of non-empty amino-acid strings. Empty sequences
are assigned 0 by the helper and are not passed to your function. Do not assume
a fixed batch size: IDiom removes empty inputs, deduplicates sequences, and can
reuse cached scores. With caching disabled, duplicates within a batch are still
scored once. Raise an informative exception for unsupported inputs instead of
returning NaN, infinity, or a shortened result list.

The helper reads one JSON request per stdin line and writes one JSON response
per stdout line. For example:

    Request:  {"sequences": ["ACDEF", "GHIKL"]}
    Response: {"scores": [24.8, 31.2]}

The numbers above illustrate the protocol, not this example's predictions.
serve() validates result counts and finite values, flushes each response, and
reports scoring exceptions as JSON errors. It redirects Python stdout to stderr
while loading and scoring, so library messages do not mix with protocol output.
Use stderr for your own logs; do not write directly to the protocol stdout.

Configuration example
---------------------
Place this in the reward YAML file, using your script path and arguments:

    reward:
      terms:
        - label: isoelectric_point
          reward:
            name: external_scorer
            cmd: "uv run --script /path/to/my_scorer.py --property isoelectric_point"
            timeout: 300
            cache_max: 100000
          shaping:
            name: gaussian
            target: 4.5
            width: 0.3
          weight: 1.0

cmd is split into command-line arguments and launched directly, not through a
shell. Quote paths containing spaces. Shell expansion, pipes, and redirection
are not interpreted. Relative paths use the training process's working
directory unless reward.cwd is set. The supplied Bash scripts change to REPO.

Runtime settings belong under reward, alongside name and cmd:

- timeout: seconds to wait for a response, including startup on the first batch
- cache_max: cached-sequence limit; the cache is cleared after exceeding it
  Set 0 for stochastic scorers such as STARLING or any changing scoring function
- maxlen: truncate sequences before scoring; 0 (the default) sends them whole
- cwd: optional working directory for the scorer process
- env: optional mapping of environment variables, added to the inherited ones
- label: optional prefix for scorer stderr, separate from the term's log label

The scorer process persists across batches. Budget GPU memory alongside the
training model, or use a separate GPU. To select the scorer's visible GPU, set
reward.env.CUDA_VISIBLE_DEVICES to a quoted device index such as "1". The
provided STARLING and ProtGPS Bash/YAML examples show their device settings.

Checking your scorer
--------------------
From the IDiom environment, run this before launching training:

    python -m idiom.train.grpo.reward.external --cmd "uv run --script /path/to/my_scorer.py"

The check prints raw scores for example sequences. Add your own sequences as
positional arguments after --cmd's value to test representative inputs. It
starts a separate process; training starts its own scorer afterward. Check the
score range and direction, then choose shaping and weights that make sense for
your objective. The check does not validate biological usefulness.
"""

import argparse

from _scorer_protocol import serve


def build():
    """Load dependencies once and return a batch scorer; serve() redirects library output to stderr."""
    # EDIT: expose the settings your scorer needs as command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", default="isoelectric_point",
                    choices=["isoelectric_point", "molecular_weight"])
    prop = ap.parse_args().property

    # EDIT: import dependencies and load any model weights once here
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    def score_batch(sequences):
        """Return one property value per non-empty sequence; let serve() report errors."""
        # EDIT: replace this calculation while preserving input order and result count
        out = []
        for seq in sequences:
            analysis = ProteinAnalysis(seq)
            out.append(analysis.isoelectric_point() if prop == "isoelectric_point"
                       else analysis.molecular_weight())
        return out

    return score_batch


if __name__ == "__main__":
    serve(build)
