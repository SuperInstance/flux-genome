"""MusicalGenome: core genome encoding a musical tradition's position in dial space.

A MusicalGenome is a vector of 25 genes (float64 in [0, 5]) organized as:
  - harmonic_genes  (indices 0–7):  control harmonic tension behavior
  - rhythmic_genes  (indices 8–15): control rhythmic complexity behavior
  - spectral_genes  (indices 16–23): control spectral density behavior
  - metadata_gene   (index 24):     encodes generation, parent info

The phenotype is a 3-tuple ``(harmonic, rhythmic, spectral)`` computed by
averaging each gene block — this maps directly to the dial position used
by the SuperInstance constraint-toolkit.

Math
----
For genes *g* ∈ [0, 5]²⁵ partitioned into three blocks of 8:

    h = (1/8) Σᵢ₌₀⁷ gᵢ
    r = (1/8) Σᵢ₌₈¹⁵ gᵢ
    s = (1/8) Σᵢ₌₁₆²³ gᵢ

The dial position is ``(h, r, s)`` ∈ [0, 5]³.
"""

from __future__ import annotations

import numpy as np


# Approximate dial centres for the 10 traditions (harmonic, rhythmic, spectral).
# These are default values; the real values come from constraint-toolkit DIAL_RANGES.
_TRADITION_CENTRES: dict[str, tuple[float, float, float]] = {
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


class MusicalGenome:
    """A genome encoding a musical tradition's position in dial space.

    Parameters
    ----------
    genes : np.ndarray
        Array of shape ``(25,)`` with values in ``[0, 5]``.
    """

    N_GENES: int = 25

    def __init__(self, genes: np.ndarray) -> None:
        genes = np.asarray(genes, dtype=np.float64).ravel()
        if genes.shape != (self.N_GENES,):
            raise ValueError(f"genes must have {self.N_GENES} elements, got {genes.shape}")
        self.genes = np.clip(genes, 0.0, 5.0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def harmonic_genes(self) -> np.ndarray:
        """Genes controlling harmonic tension (indices 0–7)."""
        return self.genes[:8]

    @property
    def rhythmic_genes(self) -> np.ndarray:
        """Genes controlling rhythmic complexity (indices 8–15)."""
        return self.genes[8:16]

    @property
    def spectral_genes(self) -> np.ndarray:
        """Genes controlling spectral density (indices 16–23)."""
        return self.genes[16:24]

    @property
    def metadata_gene(self) -> float:
        """Metadata gene encoding generation / parent info (index 24)."""
        return float(self.genes[24])

    @metadata_gene.setter
    def metadata_gene(self, value: float) -> None:
        self.genes[24] = np.clip(value, 0.0, 5.0)

    @property
    def dial_position(self) -> tuple[float, float, float]:
        """Express genes as a 3D dial position ``(harmonic, rhythmic, spectral)``.

        Computed as the mean of each 8-gene block:

            h = mean(genes[0:8])
            r = mean(genes[8:16])
            s = mean(genes[16:24])
        """
        h = float(np.mean(self.genes[:8]))
        r = float(np.mean(self.genes[8:16]))
        s = float(np.mean(self.genes[16:24]))
        return (h, r, s)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_tradition(cls, name: str) -> MusicalGenome:
        """Create genome from a known tradition's dial position.

        Parameters
        ----------
        name : str
            Tradition name (must be in the built-in catalogue).

        Returns
        -------
        MusicalGenome

        Raises
        ------
        KeyError
            If *name* is not a known tradition.
        """
        if name not in _TRADITION_CENTRES:
            raise KeyError(
                f"Unknown tradition '{name}'. "
                f"Available: {sorted(_TRADITION_CENTRES)}"
            )
        centre = _TRADITION_CENTRES[name]
        genes = np.zeros(cls.N_GENES, dtype=np.float64)
        # Fill each block around the centre with small controlled variation.
        rng = np.random.default_rng()
        for block_start, c in zip([0, 8, 16], centre):
            genes[block_start : block_start + 8] = np.clip(
                rng.normal(loc=c, scale=0.3, size=8), 0.0, 5.0
            )
        return cls(genes)

    @classmethod
    def random(cls, seed: int | None = None) -> MusicalGenome:
        """Create a random genome.

        Parameters
        ----------
        seed : int or None
            Optional RNG seed for reproducibility.

        Returns
        -------
        MusicalGenome
        """
        rng = np.random.default_rng(seed)
        return cls(rng.uniform(0.0, 5.0, size=cls.N_GENES))

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def fitness(self, target_dial: tuple[float, float, float]) -> float:
        """Distance to *target_dial* position (lower is better).

        Uses Euclidean distance in 3D dial space:

            f = √((h−hₜ)² + (r−rₜ)² + (s−sₜ)²)

        Parameters
        ----------
        target_dial : tuple[float, float, float]
            Target ``(harmonic, rhythmic, spectral)`` dial values.

        Returns
        -------
        float
        """
        pos = self.dial_position
        return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(pos, target_dial))))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        h, r, s = self.dial_position
        return f"MusicalGenome(dial=({h:.2f}, {r:.2f}, {s:.2f}))"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MusicalGenome):
            return NotImplemented
        return np.array_equal(self.genes, other.genes)

    def copy(self) -> MusicalGenome:
        """Return an independent copy."""
        return MusicalGenome(self.genes.copy())
