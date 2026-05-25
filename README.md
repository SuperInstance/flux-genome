# flux-genome

Genetic algorithm framework for evolving musical traditions in dial space.

## Overview

`flux-genome` encodes musical traditions as 25-gene chromosomes and evolves them toward target positions in a 3D dial space (harmonic tension × rhythmic complexity × spectral density).

### Gene structure

| Block | Genes | Controls |
|-------|-------|----------|
| Harmonic | 0–7 | Harmonic tension |
| Rhythmic | 8–15 | Rhythmic complexity |
| Spectral | 16–23 | Spectral density |
| Metadata | 24 | Generation/parent info |

The **dial position** is the mean of each 8-gene block → `(h, r, s)` ∈ [0, 5]³.

## Install

```bash
pip install flux-genome
```

## Quick start

```python
from flux_genome import MusicalGenome, GeneticAlgorithm

# Create a random genome
genome = MusicalGenome.random(seed=42)
print(genome.dial_position)  # (h, r, s)

# Create from a known tradition
jazz = MusicalGenome.from_tradition("Jazz")

# Evolve toward a target dial position
ga = GeneticAlgorithm(population_size=100, mutation_rate=0.1)
ga.initialize(target_dial=(3.0, 3.0, 3.0), seed=0)
ga.evolve(n_generations=50)
print(ga.best().dial_position)
```

## Crossover operators

```python
from flux_genome import uniform_crossover, arithmetic_crossover, blend_crossover

child = uniform_crossover(parent_a, parent_b, seed=42)
child = arithmetic_crossover(parent_a, parent_b, weight=0.5)
child = blend_crossover(parent_a, parent_b, alpha=0.5, seed=42)
```

## Mutation operators

```python
from flux_genome import gaussian_mutation, uniform_mutation, inversion_mutation

mutated = gaussian_mutation(genome, rate=0.1, sigma=0.5)
mutated = uniform_mutation(genome, rate=0.1)
mutated = inversion_mutation(genome)
```

## Fitness functions

```python
from flux_genome import dial_distance, novelty_score, conservation_score

dist = dial_distance(genome, target=(2.5, 2.5, 2.5))
nov = novelty_score(genome, population, k=5)
con = conservation_score(genome, ancestor)
```

## Tradition DNA

10 pre-built tradition genomes: Jazz, Classical, Rock, Blues, Electronic, Hindustani, Gamelan, Gagaku, WestAfrican, FreeImprovisation.

```python
from flux_genome import TRADITION_GENOMES, encode_tradition, decode_tradition

jazz = TRADITION_GENOMES["Jazz"]
custom = encode_tradition("MyStyle", dial_center=(2.0, 3.0, 1.5))
info = decode_tradition(jazz)
```

## Evolution log

```python
ga = GeneticAlgorithm()
ga.initialize(target_dial=(2.5, 2.5, 2.5))
ga.evolve(n_generations=100)

records = ga.log.to_records()
best = ga.log.best_ever()
```

## Dependencies

- numpy ≥ 1.24
- scipy ≥ 1.10

## License

MIT
