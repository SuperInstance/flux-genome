"""Mutation operators for MusicalGenome.

Each operator takes a genome and returns a mutated copy.

Operators
---------
- gaussian_mutation: add N(0, σ) noise to each gene with probability *rate*
- uniform_mutation: replace genes with uniform random values with probability *rate*
- inversion_mutation: reverse a contiguous subsequence of genes
"""

from __future__ import annotations

import numpy as np

from .genome import MusicalGenome


def gaussian_mutation(
    genome: MusicalGenome,
    rate: float = 0.1,
    sigma: float = 0.5,
    seed: int | None = None,
) -> MusicalGenome:
    """Gaussian mutation: add normal noise to selected genes.

    For each gene *i*, with probability *rate*:

        gene[i] ← gene[i] + N(0, σ)

    Then clamp to ``[0, 5]``.

    Parameters
    ----------
    genome : MusicalGenome
    rate : float
        Per-gene mutation probability in ``[0, 1]``.
    sigma : float
        Standard deviation of the Gaussian perturbation.
    seed : int or None

    Returns
    -------
    MusicalGenome
    """
    rng = np.random.default_rng(seed)
    genes = genome.genes.copy()
    mask = rng.random(len(genes)) < rate
    noise = rng.normal(0.0, sigma, size=len(genes))
    genes[mask] += noise[mask]
    return MusicalGenome(genes)


def uniform_mutation(
    genome: MusicalGenome,
    rate: float = 0.1,
    seed: int | None = None,
) -> MusicalGenome:
    """Uniform mutation: replace selected genes with random values in [0, 5].

    For each gene *i*, with probability *rate*:

        gene[i] ~ Uniform(0, 5)

    Parameters
    ----------
    genome : MusicalGenome
    rate : float
        Per-gene mutation probability.
    seed : int or None

    Returns
    -------
    MusicalGenome
    """
    rng = np.random.default_rng(seed)
    genes = genome.genes.copy()
    mask = rng.random(len(genes)) < rate
    genes[mask] = rng.uniform(0.0, 5.0, size=int(mask.sum()))
    return MusicalGenome(genes)


def inversion_mutation(
    genome: MusicalGenome,
    seed: int | None = None,
) -> MusicalGenome:
    """Inversion mutation: reverse a random contiguous subsequence.

    Pick two points *i* < *j* uniformly, then:

        gene[i:j+1] ← gene[i:j+1][::-1]

    Parameters
    ----------
    genome : MusicalGenome
    seed : int or None

    Returns
    -------
    MusicalGenome
    """
    rng = np.random.default_rng(seed)
    genes = genome.genes.copy()
    n = len(genes)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    genes[i : j + 1] = genes[i : j + 1][::-1]
    return MusicalGenome(genes)
