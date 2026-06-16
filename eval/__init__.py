"""Evaluation suite: compute metrics (numbers/tables) from sequences or a model.

The reusable, testable *computation* layer — sparrow sequence features, distributional distances,
generation validity, held-out perplexity, novelty, disorder. No plotting: the manuscript figures
(top-level ``analysis/``) import these so the CLI and figures compute identically. ``python -m eval.run``
(``run.main``) is the operator entry point.
"""
