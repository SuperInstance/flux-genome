"""Tests for GeneticAlgorithm."""

import pytest

from flux_genome.population import GeneticAlgorithm
from flux_genome.genome import MusicalGenome


class TestGeneticAlgorithm:
    def test_initialize(self):
        ga = GeneticAlgorithm(population_size=20)
        ga.initialize(target_dial=(2.5, 2.5, 2.5), seed=0)
        assert len(ga.population) == 20

    def test_evolve_improves(self):
        ga = GeneticAlgorithm(population_size=30, mutation_rate=0.15)
        ga.initialize(target_dial=(2.5, 2.5, 2.5), seed=0)
        ga.evolve(n_generations=20)
        best = ga.best()
        # Best should be reasonably close after 20 gens.
        assert best.fitness((2.5, 2.5, 2.5)) < 3.0

    def test_evolve_without_init_raises(self):
        ga = GeneticAlgorithm()
        with pytest.raises(RuntimeError):
            ga.evolve()

    def test_best_returns_genome(self):
        ga = GeneticAlgorithm(population_size=10)
        ga.initialize(seed=42)
        b = ga.best()
        assert isinstance(b, MusicalGenome)

    def test_log_records(self):
        ga = GeneticAlgorithm(population_size=15)
        ga.initialize(seed=1)
        records = ga.evolve(n_generations=5)
        assert len(records) == 6  # 5 + final
        assert records[0]["generation"] == 0

    def test_best_ever(self):
        ga = GeneticAlgorithm(population_size=15)
        ga.initialize(seed=1)
        ga.evolve(n_generations=10)
        be = ga.log.best_ever()
        assert be is not None
        assert be.best_fitness >= 0
