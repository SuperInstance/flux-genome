"""Tests for crossover operators."""

import numpy as np
import pytest

from flux_genome.genome import MusicalGenome
from flux_genome.crossover import uniform_crossover, arithmetic_crossover, blend_crossover


@pytest.fixture
def parents():
    a = MusicalGenome(np.full(25, 1.0))
    b = MusicalGenome(np.full(25, 4.0))
    return a, b


class TestUniformCrossover:
    def test_child_genes_from_parents(self, parents):
        a, b = parents
        child = uniform_crossover(a, b, seed=42)
        assert child.genes.shape == (25,)
        # Every gene should be either 1.0 or 4.0
        for g in child.genes:
            assert g == pytest.approx(1.0) or g == pytest.approx(4.0)

    def test_reproducible(self, parents):
        a, b = parents
        c1 = uniform_crossover(a, b, seed=10)
        c2 = uniform_crossover(a, b, seed=10)
        assert c1 == c2


class TestArithmeticCrossover:
    def test_midpoint(self, parents):
        a, b = parents
        child = arithmetic_crossover(a, b, weight=0.5)
        np.testing.assert_allclose(child.genes, 2.5)

    def test_weight_one_gives_parent_a(self, parents):
        a, b = parents
        child = arithmetic_crossover(a, b, weight=1.0)
        assert child == a


class TestBlendCrossover:
    def test_child_in_range(self, parents):
        a, b = parents
        child = blend_crossover(a, b, alpha=0.5, seed=5)
        assert child.genes.shape == (25,)
        # With a=1, b=4, alpha=0.5: lo=1-1.5=-0.5, hi=4+1.5=5.5, clamped to [0,5]
        assert child.genes.min() >= 0.0
        assert child.genes.max() <= 5.0

    def test_alpha_zero_between_parents(self, parents):
        a, b = parents
        child = blend_crossover(a, b, alpha=0.0, seed=5)
        # With alpha=0, child genes should be in [1, 4]
        assert child.genes.min() >= 1.0 - 1e-10
        assert child.genes.max() <= 4.0 + 1e-10
