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
"""Score PSpred phase-separation predictions (https://github.com/KULL-Centre/_2024_buelow_PSpred).

--target selects dG (kT), logcdil_mgml (log concentration), or cdil_mgml (mg/mL).
Lower values indicate stronger phase-separation propensity.
IDIOM_PSPRED_DIR sets the model cache (default ~/.cache/idiom/pspred), downloaded on first use.
Run with uv run --script; the scikit-learn pin is required to load the model pickles.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

from _scorer_protocol import serve

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
    """Download the predictor scripts, residue table, and models if needed; return their directory."""
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

    # Notebook pickles resolve model classes through __main__.
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


if __name__ == "__main__":
    serve(build)
