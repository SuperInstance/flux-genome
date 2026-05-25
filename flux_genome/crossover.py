"""Crossover operators for MusicalGenome recombination.

Each operator takes two parent genomes and produces one child genome,
combining genetic material in different ways.

Operators
---------
- uniform_crossover: select each gene from either parent with equal probability
- arithmetic_crossover: weighted average of parent genes
- blend_crossover: BLX-α operator sampling from an extended interval
"""

from __future__ import annotations

import numpy as np

from .genome import MusicalGenome


def uniform_crossover(
    parent_a: MusicalGenome,
    parent_b: MusicalGenome,
    seed: int | None = None,
) -> MusicalGenome:
    """Uniform crossover: each gene comes from either parent with 50% chance.

    For each locus *i* the child gene is:

        child[i] = parent_a[i]  if mask[i] == 0
                   parent_b[i]  if mask[i] == 1

    where mask ~ Bernoulli(0.5) per gene.

    Parameters
    ----------
    parent_a, parent_b : MusicalGenome
    seed : int or None

    Returns
    -------
    MusicalGenome
    """
    rng = np.random.default_rng(seed)
    mask = rng.integers(0, 2, size=MusicalGenome.N_GENES).astype(bool)
    child_genes = np.where(mask, parent_b.genes, parent_a.genes)
    return MusicalGenome(child_genes)


def arithmetic_crossover(
    parent_a: MusicalGenome,
    parent_b: MusicalGenome,
    weight: float = 0.5,
) -> MusicalGenome:
    """Arithmetic crossover: weighted average of parents.

        child[i] = weight * parent_a[i] + (1 - weight) * parent_b[i]

    Parameters
    ----------
    parent_a, parent_b : MusicalGenome
    weight : float
        Blend weight in ``[0, 1]``.  ``weight=1`` → child = parent_a.

    Returns
    -------
    MusicalGenome
    """
    child_genes = weight * parent_a.genes + (1.0 - weight) * parent_b.genes
    return MusicalGenome(child_genes)


def blend_crossover(
    parent_a: MusicalGenome,
    parent_b: MusicalGenome,
    alpha: float = 0.5,
    seed: int | None = None,
) -> MusicalGenome:
    """BLX-α crossover: sample each child gene from an extended interval.

    For each gene *i*:

        lo = min(a[i], b[i]) - α · |a[i] - b[i]|
        hi = max(a[i], b[i]) + α · |a[i] - b[i]|
        child[i] ~ Uniform(lo, hi)

    Then clamp to ``[0, 5]``.

    Parameters
    ----------
    parent_a, parent_b : MusicalGenome
    alpha : float
        Extension factor (≥ 0).  Larger α → more exploration.
    seed : int or None

    Returns
    -------
    MusicalGenome
    """
    rng = np.random.default_rng(seed)
    a, b = parent_a.genes, parent_b.genes
    lo = np.minimum(a, b) - alpha * np.abs(a - b)
    hi = np.maximum(a, b) + alpha * np.abs(a - b)
    child_genes = rng.uniform(lo, hi)
    return MusicalGenome(child_genes)
