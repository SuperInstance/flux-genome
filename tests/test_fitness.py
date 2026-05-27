"""Tests for fitness functions."""

import numpy as np
import pytest

from flux_genome.genome import MusicalGenome
from flux_genome.fitness import dial_distance, novelty_score, conservation_score


class TestDialDistance:
    def test_zero_distance(self):
        g = MusicalGenome(np.full(25, 2.0))
        assert dial_distance(g, (2.0, 2.0, 2.0)) == pytest.approx(0.0)

    def test_known_distance(self):
        g = MusicalGenome(np.full(25, 0.0))
        # dial = (0, 0, 0), target = (3, 4, 0) → dist = 5
        assert dial_distance(g, (3.0, 4.0, 0.0)) == pytest.approx(5.0)

    def test_positive(self):
        g = MusicalGenome.random(seed=42)
        d = dial_distance(g, (2.5, 2.5, 2.5))
        assert d >= 0.0


class TestNoveltyScore:
    def test_empty_population(self):
        g = MusicalGenome.random(seed=1)
        assert novelty_score(g, []) == 0.0

    def test_identical_zero(self):
        g = MusicalGenome(np.full(25, 1.0))
        pop = [MusicalGenome(np.full(25, 1.0)) for _ in range(5)]
        score = novelty_score(g, pop, k=3)
        assert score == pytest.approx(0.0)

    def test_novel_genome_higher_score(self):
        centre = MusicalGenome(np.full(25, 2.5))
        cluster = [MusicalGenome(np.full(25, 2.5)) for _ in range(10)]
        outlier = MusicalGenome(np.full(25, 4.5))
        s_cluster = novelty_score(centre, cluster, k=5)
        s_outlier = novelty_score(outlier, cluster, k=5)
        assert s_outlier > s_cluster

    def test_k_larger_than_population(self):
        g = MusicalGenome.random(seed=1)
        pop = [MusicalGenome.random(seed=i) for i in range(3)]
        # k=10 > len(pop)=3, should not error
        score = novelty_score(g, pop, k=10)
        assert score >= 0.0


class TestConservationScore:
    def test_identical_zero(self):
        g = MusicalGenome(np.full(25, 3.0))
        assert conservation_score(g, g) == pytest.approx(0.0)

    def test_negative(self):
        g1 = MusicalGenome(np.full(25, 0.0))
        g2 = MusicalGenome(np.full(25, 5.0))
        assert conservation_score(g1, g2) < 0.0

    def test_closer_higher(self):
        ancestor = MusicalGenome(np.full(25, 2.5))
        close = MusicalGenome(np.full(25, 2.6))
        far = MusicalGenome(np.full(25, 4.5))
        assert conservation_score(close, ancestor) > conservation_score(far, ancestor)
