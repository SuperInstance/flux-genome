"""Tests for EvolutionLog."""

import numpy as np
import pytest

from flux_genome.genome import MusicalGenome
from flux_genome.evolution_log import EvolutionLog, GenerationRecord


@pytest.fixture
def population():
    return [MusicalGenome.random(seed=i) for i in range(10)]


class TestEvolutionLog:
    def test_empty_log(self):
        log = EvolutionLog()
        assert log.best_ever() is None
        assert log.to_records() == []

    def test_record_adds_entry(self, population):
        log = EvolutionLog()
        log.record(0, population, (2.5, 2.5, 2.5))
        assert len(log.records) == 1
        assert log.records[0].generation == 0

    def test_record_fields(self, population):
        log = EvolutionLog()
        log.record(0, population, (2.5, 2.5, 2.5))
        rec = log.records[0]
        assert rec.best_fitness >= 0.0
        assert rec.mean_fitness >= 0.0
        assert rec.diversity >= 0.0
        assert len(rec.best_dial) == 3

    def test_best_ever(self, population):
        log = EvolutionLog()
        log.record(0, population, (2.5, 2.5, 2.5))
        # Create a better population closer to target
        better = [MusicalGenome(np.full(25, 2.5)) for _ in range(10)]
        log.record(1, better, (2.5, 2.5, 2.5))
        be = log.best_ever()
        assert be is not None
        assert be.generation == 1
        assert be.best_fitness == pytest.approx(0.0)

    def test_to_records_format(self, population):
        log = EvolutionLog()
        log.record(0, population, (2.5, 2.5, 2.5))
        recs = log.to_records()
        assert len(recs) == 1
        r = recs[0]
        assert "generation" in r
        assert "best_fitness" in r
        assert "mean_fitness" in r
        assert "diversity" in r
        assert "best_dial" in r

    def test_single_individual_diversity_zero(self):
        pop = [MusicalGenome(np.full(25, 2.5))]
        log = EvolutionLog()
        log.record(0, pop, (2.5, 2.5, 2.5))
        assert log.records[0].diversity == 0.0

    def test_diversity_positive_for_varied_pop(self, population):
        log = EvolutionLog()
        log.record(0, population, (2.5, 2.5, 2.5))
        assert log.records[0].diversity > 0.0

    def test_multiple_generations(self, population):
        log = EvolutionLog()
        for gen in range(5):
            log.record(gen, population, (2.5, 2.5, 2.5))
        assert len(log.records) == 5
        assert log.to_records()[-1]["generation"] == 4


class TestGenerationRecord:
    def test_fields(self):
        rec = GenerationRecord(
            generation=0,
            best_fitness=1.0,
            mean_fitness=2.0,
            diversity=0.5,
            best_dial=(1.0, 2.0, 3.0),
        )
        assert rec.generation == 0
        assert rec.best_fitness == 1.0
        assert rec.best_dial == (1.0, 2.0, 3.0)
