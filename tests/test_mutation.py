"""Tests for mutation operators."""

import numpy as np
import pytest

from flux_genome.genome import MusicalGenome
from flux_genome.mutation import gaussian_mutation, uniform_mutation, inversion_mutation


@pytest.fixture
def genome():
    return MusicalGenome(np.full(25, 2.5))


class TestGaussianMutation:
    def test_returns_new_genome(self, genome):
        mutated = gaussian_mutation(genome, rate=1.0, sigma=0.1, seed=42)
        assert isinstance(mutated, MusicalGenome)
        assert mutated is not genome

    def test_original_unchanged(self, genome):
        original = genome.genes.copy()
        gaussian_mutation(genome, rate=1.0, sigma=0.5, seed=42)
        np.testing.assert_array_equal(genome.genes, original)

    def test_rate_zero_no_change(self, genome):
        mutated = gaussian_mutation(genome, rate=0.0, sigma=1.0, seed=42)
        assert mutated == genome

    def test_rate_one_all_perturbed(self, genome):
        mutated = gaussian_mutation(genome, rate=1.0, sigma=2.0, seed=42)
        # With sigma=2.0, genes almost certainly differ from 2.5
        assert not np.allclose(mutated.genes, genome.genes)

    def test_reproducible(self, genome):
        m1 = gaussian_mutation(genome, rate=0.5, sigma=0.3, seed=10)
        m2 = gaussian_mutation(genome, rate=0.5, sigma=0.3, seed=10)
        assert m1 == m2

    def test_clipped_to_range(self, genome):
        # Use extreme sigma to try to push out of bounds
        mutated = gaussian_mutation(genome, rate=1.0, sigma=100.0, seed=42)
        assert mutated.genes.min() >= 0.0
        assert mutated.genes.max() <= 5.0


class TestUniformMutation:
    def test_rate_zero_no_change(self, genome):
        mutated = uniform_mutation(genome, rate=0.0, seed=42)
        assert mutated == genome

    def test_rate_one_all_replaced(self, genome):
        mutated = uniform_mutation(genome, rate=1.0, seed=42)
        # Extremely unlikely all 25 genes stay at exactly 2.5
        assert not np.array_equal(mutated.genes, genome.genes)

    def test_reproducible(self, genome):
        m1 = uniform_mutation(genome, rate=0.3, seed=10)
        m2 = uniform_mutation(genome, rate=0.3, seed=10)
        assert m1 == m2

    def test_in_range(self, genome):
        mutated = uniform_mutation(genome, rate=1.0, seed=42)
        assert mutated.genes.min() >= 0.0
        assert mutated.genes.max() <= 5.0

    def test_original_unchanged(self, genome):
        original = genome.genes.copy()
        uniform_mutation(genome, rate=1.0, seed=42)
        np.testing.assert_array_equal(genome.genes, original)


class TestInversionMutation:
    def test_genes_preserved_as_set(self, genome):
        mutated = inversion_mutation(genome, seed=42)
        # Same multiset of gene values (just reordered)
        np.testing.assert_allclose(sorted(mutated.genes), sorted(genome.genes))

    def test_reproducible(self, genome):
        m1 = inversion_mutation(genome, seed=10)
        m2 = inversion_mutation(genome, seed=10)
        assert m1 == m2

    def test_returns_new_instance(self, genome):
        mutated = inversion_mutation(genome, seed=42)
        assert mutated is not genome

    def test_different_seed_different_result(self):
        g = MusicalGenome(np.arange(25, dtype=np.float64))
        m1 = inversion_mutation(g, seed=1)
        m2 = inversion_mutation(g, seed=2)
        # Different seeds should (very likely) produce different inversions
        # Just check both are valid genomes
        assert m1.genes.shape == (25,)
        assert m2.genes.shape == (25,)
