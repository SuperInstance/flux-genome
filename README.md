# flux-genome

> Genetic evolution engine for exploring the unexplored 82% of musical dial space

Part of the [SuperInstance](https://github.com/SuperInstance) music constraint theory ecosystem. Uses genetic algorithms to breed novel musical traditions by evolving 25-gene genomes across three dial axes: Harmonic Tension, Rhythmic Complexity, and Spectral Density.

## What It Does

Most music humanity has ever made occupies a tiny fraction of the available dial space. Jazz sits in one corner, gamelan in another, Western classical somewhere in between — but the vast majority of possible musical configurations remain unexplored. **flux-genome** treats musical traditions as positions on dials and uses genetic algorithms to systematically evolve into the unknown.

Each organism is a `MusicalGenome` — a 25-gene vector encoding a complete musical identity. Genes are grouped into three chromosomes (harmonic, rhythmic, spectral) that map directly to the SuperInstance dial model. The `GeneticAlgorithm` engine runs populations through selection, crossover, and mutation, logging every generation for reproducibility.

## Key Features

- **25-gene MusicalGenome** — complete DNA encoding of a musical tradition across 3 chromosomes
- **3 crossover operators** — single-point, uniform, and blend crossover for diverse offspring
- **3 mutation operators** — Gaussian perturbation, tradition boundary snaps, and random gene resets
- **Tradition DNA encoding** — pre-encoded genomes for known traditions (jazz, blues, gamelan, etc.)
- **Tournament selection** — configurable tournament size for selection pressure tuning
- **EvolutionLog** — full lineage tracking with generation-by-generation snapshots
- **Fitness scoring** — custom fitness functions that evaluate genomes against dial-space targets
- **27 tests** — comprehensive test suite covering all operators and edge cases

## Installation

```bash
git clone https://github.com/SuperInstance/flux-genome.git
cd flux-genome
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick Start

### Create and evolve a population

```python
from flux_genome.genome import MusicalGenome
from flux_genome.genetic_algorithm import GeneticAlgorithm
from flux_genome.operators.crossover import single_point_crossover, uniform_crossover
from flux_genome.operators.mutation import gaussian_mutation, boundary_mutation

# Seed population from known traditions
from flux_genome.tradition_dna import TRADITION_DNA

jazz = MusicalGenome.from_dict(TRADITION_DNA["jazz"])
gamelan = MusicalGenome.from_dict(TRADITION_DNA["gamelan"])

# Run evolution
ga = GeneticAlgorithm(
    population_size=100,
    crossover_fn=uniform_crossover,
    mutation_fn=gaussian_mutation,
    mutation_rate=0.15,
    tournament_size=5,
)

result = ga.evolve(
    seed_genomes=[jazz, gamelan],
    generations=500,
    fitness_fn=my_fitness_function,
)

print(f"Best fitness: {result.best_fitness}")
print(f"Best genome: {result.best_genome}")
```

### Inspect evolution history

```python
log = result.evolution_log
for gen in log.generations:
    print(f"Gen {gen.number}: best={gen.best_fitness:.4f} avg={gen.avg_fitness:.4f}")
```

## Architecture

```
flux_genome/
├── genome.py              # MusicalGenome (25-gene vector, 3 chromosomes)
├── genetic_algorithm.py   # GeneticAlgorithm engine + EvolutionLog
├── population.py          # Population management, statistics
├── fitness.py             # Fitness evaluation framework
├── tradition_dna.py       # Pre-encoded tradition genomes
└── operators/
    ├── crossover.py       # single_point, uniform, blend
    └── mutation.py        # gaussian, boundary, random_reset
```

### The 25-Gene Model

Genes are organized into three chromosomes that mirror the SuperInstance dial axes:

| Chromosome | Genes | Dial Axis |
|---|---|---|
| Harmonic | 0–8 | Harmonic Tension |
| Rhythmic | 9–16 | Rhythmic Complexity |
| Spectral | 17–24 | Spectral Density |

Each gene is a float in `[0.0, 1.0]`, representing a normalized position on its respective dial.

## API Reference

### `MusicalGenome`

```python
genome = MusicalGenome(genes=[...])           # 25 floats
genome = MusicalGenome.random()                # Random valid genome
genome = MusicalGenome.from_dict(tradition)    # From tradition DNA dict
genome.to_dict()                               # Serialize
genome.hammonic_genes                          # Genes 0–8
genome.rhythmic_genes                          # Genes 9–16
genome.spectral_genes                          # Genes 17–24
```

### `GeneticAlgorithm`

```python
ga = GeneticAlgorithm(
    population_size=100,
    crossover_fn=uniform_crossover,
    mutation_fn=gaussian_mutation,
    mutation_rate=0.1,
    tournament_size=5,
    elitism=2,
)
result = ga.evolve(seed_genomes, generations, fitness_fn)
```

### `EvolutionLog`

```python
result.evolution_log.generations    # List[GenerationRecord]
result.evolution_log.best_genome    # Best genome across all generations
result.evolution_log.best_fitness   # Best fitness score
```

## Testing

```bash
pytest                    # Run all 27 tests
pytest -v                 # Verbose output
pytest --cov=flux_genome  # With coverage
```

## Related Repos

- [**flux-hyperbolic**](https://github.com/SuperInstance/flux-hyperbolic) — Hyperbolic geometry for tradition embedding and hierarchy
- [**constraint-toolkit**](https://github.com/SuperInstance/constraint-toolkit) — Core constraint satisfaction engine
- [**constraint-dsl**](https://github.com/SuperInstance/constraint-dsl) — YAML DSL for defining constraint pipelines
- [**superinstance-live**](https://github.com/SuperInstance/superinstance-live) — Live session controller using evolved genomes
- [**flux-ffi**](https://github.com/SuperInstance/flux-ffi) — FFI bindings for the shared LLVM backend

## License

MIT
