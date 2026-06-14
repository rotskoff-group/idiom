"""Figure-plumbing test (CPU-only): style loads, save_fig writes named PDF to $IDIOM_FIG_DIR."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from analysis.figures._style import COLORS, fig_dir, save_fig, use_style


def test_save_fig_writes_named_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("IDIOM_FIG_DIR", str(tmp_path))
    use_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4], color=COLORS["red"])
    ax.set_xlabel("x")
    path = save_fig(fig, "demo", subdir="si_figs")
    # output path matches the LaTeX \includegraphics path; file is written.
    assert path == tmp_path / "si_figs" / "demo.pdf"
    assert path.exists() and path.stat().st_size > 0


def test_fig_dir_requires_env(monkeypatch):
    monkeypatch.delenv("IDIOM_FIG_DIR", raising=False)
    with pytest.raises(RuntimeError, match="IDIOM_FIG_DIR"):
        fig_dir()
