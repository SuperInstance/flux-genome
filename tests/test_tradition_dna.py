"""Tests for tradition_dna module."""

import numpy as np
import pytest

from flux_genome.genome import MusicalGenome
from flux_genome.tradition_dna import (
    TRADITION_GENOMES,
    TRADITION_NAMES,
    encode_tradition,
    decode_tradition,
)


class TestTraditionGenomes:
    def test_all_traditions_present(self):
        for name in TRADITION_NAMES:
            assert name in TRADITION_GENOMES

    def test_all_are_genomes(self):
        for name, genome in TRADITION_GENOMES.items():
            assert isinstance(genome, MusicalGenome)

    def test_deterministic(self):
        for name in TRADITION_NAMES:
            g1 = TRADITION_GENOMES[name]
            # Re-import won't change; but let's just verify the dict exists and is stable
            assert g1.genes.shape == (25,)


class TestEncodeTradition:
    def test_basic_encoding(self):
        g = encode_tradition("TestTrad", (2.0, 3.0, 4.0), seed=42)
        assert isinstance(g, MusicalGenome)
        assert g.genes.shape == (25,)

    def test_genes_in_range(self):
        g = encode_tradition("Test", (2.5, 2.5, 2.5), dial_spread=10.0, seed=42)
        assert g.genes.min() >= 0.0
        assert g.genes.max() <= 5.0

    def test_reproducible(self):
        g1 = encode_tradition("X", (1.0, 2.0, 3.0), seed=7)
        g2 = encode_tradition("X", (1.0, 2.0, 3.0), seed=7)
        assert g1 == g2

    def test_dial_position_near_centre(self):
        dials = []
        for seed in range(100):
            g = encode_tradition("T", (3.0, 3.0, 3.0), dial_spread=0.1, seed=seed)
            dials.append(g.dial_position)
        mean_h = np.mean([d[0] for d in dials])
        assert abs(mean_h - 3.0) < 0.1


class TestDecodeTradition:
    def test_output_keys(self):
        g = MusicalGenome.random(seed=1)
        result = decode_tradition(g)
        assert "dial_position" in result
        assert "harmonic_genes" in result
        assert "rhythmic_genes" in result
        assert "spectral_genes" in result
        assert "metadata" in result

    def test_dial_position_matches(self):
        g = MusicalGenome.random(seed=1)
        result = decode_tradition(g)
        h, r, s = result["dial_position"]
        h2, r2, s2 = g.dial_position
        assert h == pytest.approx(h2)
        assert r == pytest.approx(r2)
        assert s == pytest.approx(s2)

    def test_gene_blocks_correct_length(self):
        g = MusicalGenome.random(seed=1)
        result = decode_tradition(g)
        assert len(result["harmonic_genes"]) == 8
        assert len(result["rhythmic_genes"]) == 8
        assert len(result["spectral_genes"]) == 8
