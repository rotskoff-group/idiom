"""Public wrappers for IDiom models and sparse autoencoders."""

from idiom.api.idiom import IDiom
from idiom.api.idiom_sae import IDiomSAE

__all__ = ["IDiom", "IDiomSAE"]


def main(argv: list[str] | None = None) -> None:
    """Run the generation CLI through the original idiom.api entrypoint."""
    from idiom.api.cli import main as cli_main

    cli_main(argv)
