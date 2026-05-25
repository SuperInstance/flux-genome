"""Encode/decode musical traditions as MusicalGenome instances.

TRADITION_GENOMES provides pre-built genomes for the 10 traditions.
``encode_tradition`` and ``decode_tradition`` convert between dial
specifications and genome representations.

The encoding scheme places the dial centre as the mean of each 8-gene
block, with per-gene variance proportional to the tradition's spread.
"""

from __future__ import annotations

import numpy as np

from .genome import MusicalGenome

# Default spread (standard deviation) for gene generation around dial centres.
_DEFAULT_SPREAD: float = 0.3

# Tradition names recognised by the catalogue.
TRADITION_NAMES: list[str] = [
    "Jazz",
    "Classical",
    "Rock",
    "Blues",
    "Electronic",
    "Hindustani",
    "Gamelan",
    "Gagaku",
    "WestAfrican",
    "FreeImprovisation",
]

# Dial centres (harmonic, rhythmic, spectral) for each tradition.
_DIAL_CENTRES: dict[str, tuple[float, float, float]] = {
    "Jazz": (3.2, 2.8, 2.5),
    "Classical": (1.8, 1.2, 1.5),
    "Rock": (3.5, 3.8, 3.0),
    "Blues": (3.0, 2.5, 2.0),
    "Electronic": (3.8, 4.0, 4.5),
    "Hindustani": (2.5, 3.2, 1.8),
    "Gamelan": (2.0, 3.5, 2.2),
    "Gagaku": (1.5, 1.8, 1.0),
    "WestAfrican": (2.8, 4.2, 2.8),
    "FreeImprovisation": (4.0, 3.5, 3.8),
}


def encode_tradition(
    name: str,
    dial_center: tuple[float, float, float],
    dial_spread: float = _DEFAULT_SPREAD,
    seed: int | None = None,
) -> MusicalGenome:
    """Encode a tradition as a MusicalGenome with controlled variation.

    Each of the three 8-gene blocks is sampled from
    N(dial_center_component, dial_spread) then clamped to [0, 5].

    Parameters
    ----------
    name : str
        Tradition name (stored in metadata for reference).
    dial_center : tuple[float, float, float]
        ``(harmonic, rhythmic, spectral)`` centre.
    dial_spread : float
        Per-gene standard deviation.
    seed : int or None

    Returns
    -------
    MusicalGenome
    """
    rng = np.random.default_rng(seed)
    genes = np.zeros(MusicalGenome.N_GENES, dtype=np.float64)
    for block_start, c in zip([0, 8, 16], dial_center):
        genes[block_start : block_start + 8] = np.clip(
            rng.normal(loc=c, scale=dial_spread, size=8), 0.0, 5.0
        )
    # Use a hash of the name for metadata gene (deterministic marker).
    genes[24] = float(hash(name) % 500) / 100.0  # in [0, 5]
    return MusicalGenome(genes)


def decode_tradition(genome: MusicalGenome) -> dict:
    """Decode a genome into a tradition-like summary.

    Returns
    -------
    dict
        Keys: ``"dial_position"``, ``"harmonic_genes"``,
        ``"rhythmic_genes"``, ``"spectral_genes"``, ``"metadata"``.
    """
    return {
        "dial_position": genome.dial_position,
        "harmonic_genes": genome.harmonic_genes.tolist(),
        "rhythmic_genes": genome.rhythmic_genes.tolist(),
        "spectral_genes": genome.spectral_genes.tolist(),
        "metadata": genome.metadata_gene,
    }


# Pre-built genomes (deterministic seeds for reproducibility).
TRADITION_GENOMES: dict[str, MusicalGenome] = {
    name: encode_tradition(name, _DIAL_CENTRES[name], seed=i)
    for i, name in enumerate(TRADITION_NAMES)
}
