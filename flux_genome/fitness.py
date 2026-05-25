"""Fitness functions for evaluating MusicalGenome quality.

Functions
---------
- dial_distance: Euclidean distance to a target dial position
- novelty_score: average distance to nearest neighbours (encourages diversity)
- conservation_score: similarity to an ancestor genome (encourages heritage)
"""

from __future__ import annotations

import numpy as np

from .genome import MusicalGenome


def dial_distance(
    genome: MusicalGenome,
    target: tuple[float, float, float],
) -> float:
    """Euclidean distance in 3D dial space to *target*.

        f = √((h − hₜ)² + (r − rₜ)² + (s − sₜ)²)

    Parameters
    ----------
    genome : MusicalGenome
    target : tuple[float, float, float]
        ``(harmonic, rhythmic, spectral)`` target.

    Returns
    -------
    float
    """
    return genome.fitness(target)


def novelty_score(
    genome: MusicalGenome,
    population: list[MusicalGenome],
    k: int = 5,
) -> float:
    """Novelty score: mean distance to *k* nearest neighbours.

    Higher scores indicate more novel (diverse) individuals.

        novelty = (1/k) Σⱼ₌₁ᵏ d(genome, nearest_j)

    Parameters
    ----------
    genome : MusicalGenome
    population : list[MusicalGenome]
        Reference population (may include *genome* itself).
    k : int
        Number of nearest neighbours to consider.

    Returns
    -------
    float
    """
    if not population:
        return 0.0
    dists = np.array([_euclidean(genome.genes, other.genes) for other in population])
    dists.sort()
    k = min(k, len(dists))
    return float(np.mean(dists[:k]))


def conservation_score(
    genome: MusicalGenome,
    ancestor: MusicalGenome,
) -> float:
    """Conservation score: negative Euclidean distance to an ancestor.

    Higher (less negative) means more conserved:

        conservation = −‖genes − ancestor_genes‖₂

    Parameters
    ----------
    genome : MusicalGenome
    ancestor : MusicalGenome
        Ancestral genome to compare against.

    Returns
    -------
    float
    """
    return -float(np.linalg.norm(genome.genes - ancestor.genes))


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
