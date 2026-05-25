"""Tests for MusicalGenome."""

import numpy as np
import pytest

from flux_genome.genome import MusicalGenome


class TestMusicalGenomeCreation:
    def test_basic_creation(self):
        genes = np.ones(25) * 2.5
        g = MusicalGenome(genes)
        assert g.genes.shape == (25,)
        np.testing.assert_allclose(g.genes, 2.5)

    def test_clipping(self):
        g = MusicalGenome(np.full(25, 10.0))
        assert g.genes.max() <= 5.0
        g2 = MusicalGenome(np.full(25, -5.0))
        assert g2.genes.min() >= 0.0

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            MusicalGenome(np.ones(10))

    def test_random(self):
        g = MusicalGenome.random(seed=42)
        assert g.genes.shape == (25,)
        assert g.genes.min() >= 0.0
        assert g.genes.max() <= 5.0

    def test_random_reproducible(self):
        a = MusicalGenome.random(seed=7)
        b = MusicalGenome.random(seed=7)
        assert a == b

    def test_from_tradition(self):
        g = MusicalGenome.from_tradition("Jazz")
        assert g.genes.shape == (25,)

    def test_from_unknown_tradition_raises(self):
        with pytest.raises(KeyError):
            MusicalGenome.from_tradition("FakeTradition")


class TestDialPosition:
    def test_uniform_genes(self):
        g = MusicalGenome(np.full(25, 3.0))
        h, r, s = g.dial_position
        assert h == pytest.approx(3.0)
        assert r == pytest.approx(3.0)
        assert s == pytest.approx(3.0)

    def test_from_tradition_near_centre(self):
        # Run many times; the mean should be close to the centre.
        dials = []
        for seed in range(50):
            g = MusicalGenome.from_tradition("Classical")
            dials.append(g.dial_position)
        mean_h = np.mean([d[0] for d in dials])
        assert abs(mean_h - 1.8) < 0.3


class TestFitness:
    def test_zero_distance(self):
        g = MusicalGenome(np.full(25, 2.0))
        # dial position will be (2, 2, 2) → distance to (2, 2, 2) = 0
        assert g.fitness((2.0, 2.0, 2.0)) == pytest.approx(0.0)

    def test_known_distance(self):
        g = MusicalGenome(np.full(25, 0.0))
        # dial = (0, 0, 0), target = (3, 4, 0) → dist = 5
        assert g.fitness((3.0, 4.0, 0.0)) == pytest.approx(5.0)


class TestProperties:
    def test_gene_blocks(self):
        g = MusicalGenome.random(seed=1)
        assert len(g.harmonic_genes) == 8
        assert len(g.rhythmic_genes) == 8
        assert len(g.spectral_genes) == 8

    def test_metadata_gene(self):
        g = MusicalGenome.random(seed=1)
        assert 0.0 <= g.metadata_gene <= 5.0

    def test_metadata_setter(self):
        g = MusicalGenome.random(seed=1)
        g.metadata_gene = 4.0
        assert g.metadata_gene == pytest.approx(4.0)

    def test_copy(self):
        g = MusicalGenome.random(seed=1)
        c = g.copy()
        assert g == c
        c.genes[0] = 99
        assert g.genes[0] != 99
