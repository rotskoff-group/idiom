# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "scikit-learn==1.6.*",
#   "MDAnalysis",
#   "biopython",
#   "numba",
#   "pandas",
#   "joblib",
#   "numpy<2",
# ]
# ///
"""Score IDRs with PSpred (https://github.com/KULL-Centre/_2024_buelow_PSpred) as an external reward.

PSpred predicts the thermodynamics of homotypic phase separation from sequence -- the transfer free
energy dG in kT, and the saturation concentration c_sat in mg/mL -- with models trained on CALVADOS
coarse-grained simulations (von Bulow et al., PNAS 2025). For scale: LAF1 scores dG = -6.1 kT with
c_sat = 1.2 mg/mL, poly-GS scores dG = +0.3 kT with c_sat = 80 mg/mL.

    --target dG              transfer free energy in kT; more negative phase-separates more readily
    --target logcdil_mgml    log saturation concentration; lower phase-separates more readily
    --target cdil_mgml       the same, exponentiated to mg/mL

    reward.terms:
      - {reward: {name: scorer, cmd: "uv run --script cookbook/rewards/scorers/pspred.py --target dG"},
         shaping: {name: quadratic, target: -6.0, width: 0.3}, label: dG, weight: 1.0}

The predictor's own files (3.7 MB: two scripts, a residue table and three joblib models) are fetched
once from the project's GitHub into IDIOM_PSPRED_DIR. The scikit-learn pin matters: the models are
pickles and will not load against a different version.

Environment variables:
    IDIOM_PSPRED_DIR   where the predictor files live (default ~/.cache/idiom/pspred).
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

GITHUB_RAW = "https://raw.githubusercontent.com/KULL-Centre/_2024_buelow_PSpred/main"
FILES = {
    "sequence.py": f"{GITHUB_RAW}/scripts_colab/sequence.py",
    "predictor.py": f"{GITHUB_RAW}/scripts_colab/predictor.py",
    "residues.csv": f"{GITHUB_RAW}/data/residues.csv",
    "svr_model_nu.joblib": f"{GITHUB_RAW}/models/svr_model_nu.joblib",
    "model_dG.joblib": f"{GITHUB_RAW}/models/idrome90/mlp/dG/model.joblib",
    "model_logcdil_mgml.joblib": f"{GITHUB_RAW}/models/idrome90/mlp/logcdil_mgml/model.joblib",
}
FEATURES = ["mean_lambda", "faro", "shd", "ncpr", "fcr", "scd", "ah_ij", "nu_svr"]


def _pspred_dir() -> Path:
    """Return the predictor directory, downloading its files on first use.

    Returns:
        Path: A directory holding the two scripts, the residue table and the three models.
    """
    d = Path(os.environ.get("IDIOM_PSPRED_DIR", Path.home() / ".cache/idiom/pspred")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    missing = {name: url for name, url in FILES.items() if not (d / name).exists()}
    if missing:
        print(f"downloading PSpred model files (3.7 MB) to {d} ...", file=sys.stderr, flush=True)
        for name, url in missing.items():
            urllib.request.urlretrieve(url, d / name)
    return d


def build():
    """Parse arguments, fetch the predictor files if needed, and return the phase-separation scorer."""
    ap = argparse.ArgumentParser(description="PSpred phase-separation propensity as an IDiom reward.")
    ap.add_argument("--target", default="dG", choices=("dG", "logcdil_mgml", "cdil_mgml"),
                    help="transfer free energy (kT), log saturation concentration, or c_sat in mg/mL")
    target = ap.parse_args().target

    d = _pspred_dir()
    sys.path.insert(0, str(d))
    os.chdir(d)  # the predictor resolves residues.csv and the nu model relative to the cwd

    import joblib
    import numpy as np
    import pandas as pd
    import predictor
    from predictor import X_from_seq

    # The joblib models were pickled from a notebook, so their classes are looked up in __main__;
    # without this, loading fails with "Can't get attribute 'Model' on <module '__main__'>".
    import __main__
    for name in dir(predictor):
        if not name.startswith("_") and not hasattr(__main__, name):
            setattr(__main__, name, getattr(predictor, name))

    residues = pd.read_csv(d / "residues.csv").set_index("one")
    model_key = "logcdil_mgml" if target in ("logcdil_mgml", "cdil_mgml") else "dG"
    model = joblib.load(d / f"model_{model_key}.joblib")

    def score_batch(sequences):
        """Return the predicted quantity per sequence."""
        rows = [X_from_seq(s, FEATURES, residues=residues, charge_termini=True,
                           nu_file=str(d / "svr_model_nu.joblib")) for s in sequences]
        # the model is an ensemble of cross-validation folds; its prediction is their mean
        preds = [float(np.mean(model.predict(row))) for row in rows]
        return [float(np.exp(v)) if target == "cdil_mgml" else v for v in preds]

    return score_batch


def serve(build):
    """Drive the newline-JSON scorer protocol until stdin closes.

    build() is called once, after stdout is claimed for the protocol, and returns score_batch: a
    function mapping a list of (non-empty) residue strings to one raw score each. Doing the imports
    and model loading inside build() keeps any chatter they print off the protocol stream.

    Args:
        build (Callable[[], Callable[[list[str]], list[float]]]): Returns the batch scorer.
    """
    # This file's own directory is sys.path[0]; drop it so a scorer named after the package it wraps
    # (sparrow.py importing sparrow) resolves to the installed package, not back to itself.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    # stdout is the protocol. A library that prints on import (TensorFlow, ProtGPS, STARLING) would
    # corrupt the first response, so keep the real stdout for responses and send chatter to stderr.
    out, sys.stdout = sys.stdout, sys.stderr
    score_batch = build()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            seqs = json.loads(line)["sequences"]
            keep = [(i, s) for i, s in enumerate(seqs) if s]  # empty completions score 0.0
            values = score_batch([s for _, s in keep]) if keep else []
            scores = [0.0] * len(seqs)
            for (i, _), v in zip(keep, values):
                scores[i] = float(v)
            payload = {"scores": scores}
        except Exception as e:
            payload = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(payload), file=out, flush=True)


if __name__ == "__main__":
    serve(build)
