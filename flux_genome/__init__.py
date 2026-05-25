"""flux-genome: Genetic algorithm framework for evolving musical traditions in dial space."""

from .genome import MusicalGenome
from .population import GeneticAlgorithm
from .tradition_dna import TRADITION_GENOMES, encode_tradition, decode_tradition
from .crossover import uniform_crossover, arithmetic_crossover, blend_crossover
from .mutation import gaussian_mutation, uniform_mutation, inversion_mutation
from .fitness import dial_distance, novelty_score, conservation_score
from .evolution_log import EvolutionLog

__all__ = [
    "MusicalGenome",
    "GeneticAlgorithm",
    "TRADITION_GENOMES",
    "encode_tradition",
    "decode_tradition",
    "uniform_crossover",
    "arithmetic_crossover",
    "blend_crossover",
    "gaussian_mutation",
    "uniform_mutation",
    "inversion_mutation",
    "dial_distance",
    "novelty_score",
    "conservation_score",
    "EvolutionLog",
]
