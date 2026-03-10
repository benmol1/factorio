"""Tests for the utils module."""

import numpy as np
import pytest

from src.quality import (
    NUM_TIERS,
    QualityTier,
    create_production_matrix,
    quality_matrix,
    quality_probability,
)


class TestQualityProbability:
    """Tests for quality_probability function."""

    def test_no_downgrade(self):
        """Items should never downgrade in quality."""
        for input_tier in range(NUM_TIERS):
            for output_tier in range(input_tier):
                assert quality_probability(0.5, input_tier, output_tier) == 0

    def test_legendary_stays_legendary(self):
        """Legendary items always stay legendary."""
        assert quality_probability(0.5, QualityTier.Legendary, QualityTier.Legendary) == 1
        assert quality_probability(1.0, QualityTier.Legendary, QualityTier.Legendary) == 1

    def test_zero_quality_chance(self):
        """With 0% quality chance, items stay at their tier."""
        for tier in range(NUM_TIERS - 1):
            assert quality_probability(0, tier, tier) == 1
            for higher_tier in range(tier + 1, NUM_TIERS):
                assert quality_probability(0, tier, higher_tier) == 0

    def test_same_tier_probability(self):
        """Probability of staying at same tier is 1 - quality_chance."""
        assert quality_probability(0.25, 0, 0) == 0.75
        assert quality_probability(0.5, 1, 1) == 0.5

    def test_probabilities_sum_to_one(self):
        """All output probabilities for a given input tier should sum to 1."""
        for quality_chance in [0.1, 0.25, 0.5, 1.0]:
            for input_tier in range(NUM_TIERS):
                total = sum(
                    quality_probability(quality_chance, input_tier, output_tier) for output_tier in range(NUM_TIERS)
                )
                assert abs(total - 1.0) < 1e-10, f"Sum was {total} for tier {input_tier}"

    def test_invalid_quality_chance_raises(self):
        """Invalid quality chance should raise AssertionError."""
        with pytest.raises(AssertionError):
            quality_probability(-0.1, 0, 0)
        with pytest.raises(AssertionError):
            quality_probability(1.1, 0, 0)


class TestQualityMatrix:
    """Tests for quality_matrix function."""

    def test_matrix_shape(self):
        """Matrix should be NUM_TIERS x NUM_TIERS."""
        mat = quality_matrix(0.25)
        assert mat.shape == (NUM_TIERS, NUM_TIERS)

    def test_rows_sum_to_one(self):
        """Each row should sum to 1 (probability distribution)."""
        mat = quality_matrix(0.25)
        for row in range(NUM_TIERS):
            assert abs(mat[row].sum() - 1.0) < 1e-10

    def test_lower_triangle_zero(self):
        """Lower triangle should be zero (no downgrades)."""
        mat = quality_matrix(0.25)
        for i in range(NUM_TIERS):
            for j in range(i):
                assert mat[i, j] == 0

    def test_legendary_row(self):
        """Legendary row should be all zeros except last column."""
        mat = quality_matrix(0.25)
        expected = np.array([0, 0, 0, 0, 1])
        np.testing.assert_array_equal(mat[4], expected)


class TestCreateProductionMatrix:
    """Tests for create_production_matrix function."""

    def test_matrix_shape(self):
        """Matrix should be NUM_TIERS x NUM_TIERS."""
        params = [(0.25, 1.0)] * NUM_TIERS
        mat = create_production_matrix(params)
        assert mat.shape == (NUM_TIERS, NUM_TIERS)

    def test_production_ratio_scaling(self):
        """Production ratio should scale the quality matrix."""
        params_1x = [(0.25, 1.0)] * NUM_TIERS
        params_2x = [(0.25, 2.0)] * NUM_TIERS

        mat_1x = create_production_matrix(params_1x)
        mat_2x = create_production_matrix(params_2x)

        np.testing.assert_array_almost_equal(mat_2x, mat_1x * 2)

    def test_zero_production_row(self):
        """Row with zero production should be all zeros."""
        params = [(0.25, 1.0)] * 4 + [(0, 0)]
        mat = create_production_matrix(params)
        np.testing.assert_array_equal(mat[4], np.zeros(NUM_TIERS))

    def test_invalid_params_length_raises(self):
        """Wrong number of parameters should raise AssertionError."""
        with pytest.raises(AssertionError):
            create_production_matrix([(0.25, 1.0)] * 3)

    def test_recycler_params(self):
        """Test typical recycler parameters (25% return, 4 quality modules)."""
        quality_chance = 4 * 0.062  # 4 legendary quality modules
        production_ratio = 0.25  # Recycling returns 25%
        params = [(quality_chance, production_ratio)] * 4 + [(0, 0)]

        mat = create_production_matrix(params)

        # Row sums should equal production_ratio for recycled tiers
        for row in range(4):
            assert abs(mat[row].sum() - production_ratio) < 1e-10

        # Last row should be zeros (legendary not recycled)
        np.testing.assert_array_equal(mat[4], np.zeros(NUM_TIERS))
