"""GeneticAlgorithm: evolve musical traditions via genetic operations.

Combines selection, crossover, and mutation to optimise genomes
toward a target dial position over successive generations.
"""

from __future__ import annotations

import numpy as np

from .genome import MusicalGenome
from .crossover import blend_crossover
from .mutation import gaussian_mutation
from .evolution_log import EvolutionLog
from .tradition_dna import TRADITION_GENOMES


class GeneticAlgorithm:
    """Evolve musical traditions via genetic operations.

    Parameters
    ----------
    population_size : int
        Number of individuals per generation.
    mutation_rate : float
        Per-gene mutation probability for Gaussian mutation.
    crossover_rate : float
        Probability of applying crossover (vs copying a parent).
    tournament_size : int
        Number of individuals in tournament selection.
    """

    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.population: list[MusicalGenome] = []
        self._target: tuple[float, float, float] | None = None
        self._rng = np.random.default_rng()
        self.log = EvolutionLog()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(
        self,
        target_dial: tuple[float, float, float] = (2.5, 2.5, 2.5),
        traditions: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        """Seed the population with known traditions plus random genomes.

        Parameters
        ----------
        target_dial : tuple[float, float, float]
            Target dial for fitness evaluation.
        traditions : list[str] or None
            Tradition names to seed.  ``None`` → use all 10.
        seed : int or None
            RNG seed.
        """
        self._rng = np.random.default_rng(seed)
        self._target = target_dial
        self.population = []

        # Seed with tradition genomes.
        if traditions is None:
            traditions = list(TRADITION_GENOMES.keys())
        for name in traditions:
            if name in TRADITION_GENOMES:
                self.population.append(TRADITION_GENOMES[name].copy())

        # Fill remaining slots with random genomes.
        while len(self.population) < self.population_size:
            self.population.append(
                MusicalGenome.random(seed=int(self._rng.integers(0, 2**31)))
            )

        # Trim to exact size.
        self.population = self.population[: self.population_size]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self) -> tuple[MusicalGenome, MusicalGenome]:
        """Tournament selection: pick two parents.

        For each parent, sample *tournament_size* individuals and return
        the one with the best (lowest) fitness.

        Returns
        -------
        tuple[MusicalGenome, MusicalGenome]
        """
        def _tournament() -> MusicalGenome:
            indices = self._rng.choice(
                len(self.population), size=self.tournament_size, replace=False
            )
            candidates = [self.population[i] for i in indices]
            return min(candidates, key=lambda g: g.fitness(self._target))

        return _tournament(), _tournament()

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve(
        self,
        n_generations: int = 100,
    ) -> list[dict]:
        """Run evolution for *n_generations*.

        Parameters
        ----------
        n_generations : int

        Returns
        -------
        list[dict]
            The evolution log records.
        """
        if not self.population or self._target is None:
            raise RuntimeError("Call initialize() before evolve()")

        self.log = EvolutionLog()

        for gen in range(n_generations):
            self.log.record(gen, self.population, self._target)

            new_pop: list[MusicalGenome] = []
            # Elitism: carry forward the best individual.
            best = min(self.population, key=lambda g: g.fitness(self._target))
            new_pop.append(best.copy())

            while len(new_pop) < self.population_size:
                p1, p2 = self.select()
                if self._rng.random() < self.crossover_rate:
                    child = blend_crossover(p1, p2, alpha=0.5)
                else:
                    child = p1.copy()
                child = gaussian_mutation(
                    child, rate=self.mutation_rate, sigma=0.3
                )
                new_pop.append(child)

            self.population = new_pop[: self.population_size]

        # Final generation record.
        self.log.record(n_generations, self.population, self._target)
        return self.log.to_records()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def best(self) -> MusicalGenome:
        """Return the current best genome (lowest fitness)."""
        if not self.population:
            raise RuntimeError("Population is empty")
        return min(self.population, key=lambda g: g.fitness(self._target))
