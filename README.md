# flux-genome

> Genetic algorithm framework for evolving musical traditions in dial space.

`flux-genome` encodes musical traditions as 25-gene chromosomes and evolves them toward target positions in a 3D **dial space** (harmonic tension, rhythmic complexity, spectral density). It provides configurable crossover, mutation, and fitness operators, tournament selection with elitism, and full evolution logging — all designed to integrate with the [SuperInstance](https://github.com/SuperInstance) constraint-toolkit.

## How It Works

### MusicalGenome

A `MusicalGenome` is a vector of **25 float64 genes** clamped to `[0, 5]`, organized in four blocks:

| Block | Indices | Controls |
|---|---|---|
| Harmonic genes | 0–7 | Harmonic tension behavior |
| Rhythmic genes | 8–15 | Rhythmic complexity behavior |
| Spectral genes | 16–23 | Spectral density behavior |
| Metadata gene | 24 | Generation / parent info |

The **phenotype** is a 3-tuple dial position computed by averaging each 8-gene block:

```
h = mean(genes[0:8])
r = mean(genes[8:16])
s = mean(genes[16:24])
```

### Tradition DNA

Ten built-in traditions are encoded as genomes with known dial centres:

| Tradition | Harmonic | Rhythmic | Spectral |
|---|---|---|---|
| Jazz | 3.2 | 2.8 | 2.5 |
| Classical | 1.8 | 1.2 | 1.5 |
| Rock | 3.5 | 3.8 | 3.0 |
| Blues | 3.0 | 2.5 | 2.0 |
| Electronic | 3.8 | 4.0 | 4.5 |
| Hindustani | 2.5 | 3.2 | 1.8 |
| Gamelan | 2.0 | 3.5 | 2.2 |
| Gagaku | 1.5 | 1.8 | 1.0 |
| WestAfrican | 2.8 | 4.2 | 2.8 |
| FreeImprovisation | 4.0 | 3.5 | 3.8 |

### Genetic Operators

#### Crossover (3 operators)

- **Uniform crossover** — each gene is selected from either parent with 50% probability
- **Arithmetic crossover** — weighted average: `child = w * A + (1-w) * B`
- **BLX-α blend crossover** — samples each child gene from an extended interval `[min(a,b) - α|a-b|, max(a,b) + α|a-b|]`

#### Mutation (3 operators)

- **Gaussian mutation** — adds `N(0, σ)` noise to each gene with per-gene probability `rate`
- **Uniform mutation** — replaces selected genes with `Uniform(0, 5)` values
- **Inversion mutation** — reverses a random contiguous subsequence of genes

#### Fitness Functions

- **Dial distance** — Euclidean distance to a target dial position
- **Novelty score** — mean distance to k-nearest neighbours (encourages diversity)
- **Conservation score** — negative Euclidean distance to an ancestor genome (encourages heritage)

### Evolution Loop

`GeneticAlgorithm` combines tournament selection (configurable size), BLX-α crossover, Gaussian mutation, and **elitism** (best individual always survives). An `EvolutionLog` records per-generation statistics: best fitness, mean fitness, population diversity (mean pairwise distance), and best dial position.

## Installation

```bash
pip install flux-genome
```

Requires Python ≥ 3.10, NumPy ≥ 1.24, and SciPy ≥ 1.10.

For development:

```bash
pip install flux-genome[dev]
```

## Quick Start

### Create and inspect a genome

```python
from flux_genome import MusicalGenome

# Random genome
g = MusicalGenome.random(seed=42)
print(g)  # MusicalGenome(dial=(2.34, 2.81, 2.19))

# From a known tradition
jazz = MusicalGenome.from_tradition("Jazz")
print(jazz.dial_position)  # (~3.2, ~2.8, ~2.5)

# Fitness against a target dial
print(jazz.fitness((3.2, 2.8, 2.5)))  # ~0.0
```

### Run evolution

```python
from flux_genome import GeneticAlgorithm

ga = GeneticAlgorithm(
    population_size=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    tournament_size=3,
)
ga.initialize(target_dial=(3.0, 3.0, 3.0), seed=0)
records = ga.evolve(n_generations=50)

print(ga.best().dial_position)  # Close to (3.0, 3.0, 3.0)
print(ga.log.best_ever().best_fitness)
```

### Crossover and mutation

```python
from flux_genome import MusicalGenome
from flux_genome import uniform_crossover, arithmetic_crossover, blend_crossover
from flux_genome import gaussian_mutation, uniform_mutation, inversion_mutation

a = MusicalGenome.from_tradition("Jazz")
b = MusicalGenome.from_tradition("Classical")

# Blend two traditions
child = blend_crossover(a, b, alpha=0.5, seed=0)

# Mutate
mutated = gaussian_mutation(child, rate=0.2, sigma=0.5)
```

### Tradition encoding / decoding

```python
from flux_genome import encode_tradition, decode_tradition

genome = encode_tradition("Jazz", dial_center=(3.2, 2.8, 2.5), seed=0)
info = decode_tradition(genome)
print(info["dial_position"])   # (harmonic, rhythmic, spectral)
print(info["harmonic_genes"])  # 8 float values
```

### Evolution log analysis

```python
records = ga.log.to_records()  # list of dicts
for r in records:
    print(f"Gen {r['generation']}: best={r['best_fitness']:.4f}, "
          f"mean={r['mean_fitness']:.4f}, diversity={r['diversity']:.4f}")

best = ga.log.best_ever()
print(f"Best ever: gen {best.generation}, fitness {best.best_fitness:.4f}")
```

## API Reference

### `MusicalGenome`

| Method / Property | Description |
|---|---|
| `MusicalGenome(genes)` | Create from a 25-element array |
| `MusicalGenome.random(seed)` | Random genome |
| `MusicalGenome.from_tradition(name)` | Genome from built-in tradition catalogue |
| `.dial_position` | 3-tuple `(harmonic, rhythmic, spectral)` |
| `.harmonic_genes` | Genes 0–7 |
| `.rhythmic_genes` | Genes 8–15 |
| `.spectral_genes` | Genes 16–23 |
| `.metadata_gene` | Gene 24 (generation/parent info) |
| `.fitness(target_dial)` | Euclidean distance to target |
| `.copy()` | Independent clone |

### `GeneticAlgorithm`

| Method | Description |
|---|---|
| `GeneticAlgorithm(population_size, mutation_rate, crossover_rate, tournament_size)` | Configure the GA |
| `.initialize(target_dial, traditions, seed)` | Seed population with traditions + random genomes |
| `.evolve(n_generations)` | Run evolution, return log records |
| `.best()` | Current best genome |
| `.log` | `EvolutionLog` instance |

### Crossover Operators

| Function | Signature |
|---|---|
| `uniform_crossover(a, b, seed)` | Each gene from either parent |
| `arithmetic_crossover(a, b, weight)` | Weighted average |
| `blend_crossover(a, b, alpha, seed)` | BLX-α extended interval sampling |

### Mutation Operators

| Function | Signature |
|---|---|
| `gaussian_mutation(genome, rate, sigma, seed)` | Add N(0, σ) noise per gene |
| `uniform_mutation(genome, rate, seed)` | Replace genes with U(0, 5) |
| `inversion_mutation(genome, seed)` | Reverse a random gene segment |

### Fitness Functions

| Function | Description |
|---|---|
| `dial_distance(genome, target)` | Euclidean distance in dial space |
| `novelty_score(genome, population, k)` | Mean distance to k nearest neighbours |
| `conservation_score(genome, ancestor)` | Negative distance to ancestor |

### `EvolutionLog`

| Method | Description |
|---|---|
| `.record(generation, population, target_dial)` | Record a generation snapshot |
| `.best_ever()` | `GenerationRecord` with lowest best fitness |
| `.to_records()` | Export all records as list of dicts |

## Related Repos

- **[constraint-toolkit](https://github.com/SuperInstance/constraint-toolkit)** — Dial space definitions and constraint solving
- **[flux-hyperbolic](https://github.com/SuperInstance/flux-hyperbolic)** — Hyperbolic geometry embeddings for tradition hierarchies
- **[superinstance-live](https://github.com/SuperInstance/superinstance-live)** — Live session controller using evolved genomes
- **[plato-client](https://github.com/SuperInstance/plato-client)** — Client library for the Plato optimization backend

## License

MIT
