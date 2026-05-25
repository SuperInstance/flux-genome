"""EvolutionLog: track evolutionary history across generations.

Records per-generation statistics (best fitness, mean fitness, diversity)
and supports querying the history for analysis or visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .genome import MusicalGenome


@dataclass
class GenerationRecord:
    """Snapshot of a single generation."""

    generation: int
    best_fitness: float
    mean_fitness: float
    diversity: float  # mean pairwise distance in population
    best_dial: tuple[float, float, float]


@dataclass
class EvolutionLog:
    """Accumulates generation-by-generation evolution statistics.

    Usage::

        log = EvolutionLog()
        for gen in range(n):
            ...
            log.record(gen, population, target_dial)

        # Query
        df = log.to_records()  # list of dicts
        log.best_ever()        # GenerationRecord of best fitness
    """

    records: list[GenerationRecord] = field(default_factory=list)

    def record(
        self,
        generation: int,
        population: list[MusicalGenome],
        target_dial: tuple[float, float, float],
    ) -> None:
        """Record a generation snapshot.

        Parameters
        ----------
        generation : int
            Zero-indexed generation number.
        population : list[MusicalGenome]
            Current population.
        target_dial : tuple[float, float, float]
            Target dial for fitness evaluation.
        """
        fitnesses = [g.fitness(target_dial) for g in population]
        best_idx = int(np.argmin(fitnesses))
        best = population[best_idx]

        # Diversity: mean pairwise Euclidean distance.
        genes = np.array([g.genes for g in population])
        n = len(population)
        if n > 1:
            diffs = genes[:, None, :] - genes[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=-1))
            diversity = float(dists.sum() / (n * (n - 1)))
        else:
            diversity = 0.0

        rec = GenerationRecord(
            generation=generation,
            best_fitness=fitnesses[best_idx],
            mean_fitness=float(np.mean(fitnesses)),
            diversity=diversity,
            best_dial=best.dial_position,
        )
        self.records.append(rec)

    def best_ever(self) -> GenerationRecord | None:
        """Return the generation record with the lowest best_fitness."""
        if not self.records:
            return None
        return min(self.records, key=lambda r: r.best_fitness)

    def to_records(self) -> list[dict[str, Any]]:
        """Export all records as a list of plain dicts."""
        return [
            {
                "generation": r.generation,
                "best_fitness": r.best_fitness,
                "mean_fitness": r.mean_fitness,
                "diversity": r.diversity,
                "best_dial": r.best_dial,
            }
            for r in self.records
        ]
