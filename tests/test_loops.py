"""Tests for the recycler and assembler-recycler loop simulations."""

import numpy as np

from src.quality import BEST_QUAL_MODULE, NUM_TIERS
from src.quality.assembler_recycler_loop import (
    assembler_recycler_loop,
    create_transition_matrix,
    get_assembler_parameters,
    get_recycler_parameters,
)
from src.quality.pure_recycler_loop import recycler_loop, recycler_matrix


class TestRecyclerMatrix:
    """Tests for recycler_matrix function."""

    def test_matrix_shape(self):
        """Matrix should be NUM_TIERS x NUM_TIERS."""
        mat = recycler_matrix(0.25)
        assert mat.shape == (NUM_TIERS, NUM_TIERS)

    def test_recycled_rows_sum_to_025(self):
        """Recycled rows should sum to 0.25 (recycling return rate)."""
        mat = recycler_matrix(0.25, quality_to_keep=5)
        for row in range(4):  # First 4 tiers are recycled
            assert abs(mat[row].sum() - 0.25) < 1e-10

    def test_kept_row_is_zero(self):
        """Kept tier row should be all zeros."""
        mat = recycler_matrix(0.25, quality_to_keep=5)
        np.testing.assert_array_equal(mat[4], np.zeros(NUM_TIERS))

    def test_asteroid_crusher_production_ratio(self):
        """Asteroid crusher should use 0.8 production ratio."""
        mat = recycler_matrix(0.25, quality_to_keep=5, is_asteroid_crusher=True)
        for row in range(4):
            assert abs(mat[row].sum() - 0.8) < 1e-10


class TestRecyclerLoop:
    """Tests for recycler_loop function."""

    def test_output_shape(self):
        """Output flow vector should have NUM_TIERS elements."""
        flows, _, _ = recycler_loop(100.0, quality_chance=0.25)
        assert flows.shape == (NUM_TIERS,)

    def test_scalar_input(self):
        """Scalar input should be converted to array."""
        flows, _, _ = recycler_loop(100.0, quality_chance=0.25)
        assert isinstance(flows, np.ndarray)

    def test_array_input(self):
        """Array input should work."""
        input_vec = np.array([100.0, 0, 0, 0, 0])
        flows, _, _ = recycler_loop(input_vec, quality_chance=0.25)
        assert flows.shape == (NUM_TIERS,)

    def test_legendary_output_positive(self):
        """Should produce some legendary items."""
        flows, _, _ = recycler_loop(100.0, quality_chance=4 * BEST_QUAL_MODULE)
        assert flows[4] > 0  # Legendary tier

    def test_mass_conservation(self):
        """Total output should equal input * production_ratio for kept tier."""
        # With quality_to_keep=5, only legendary is kept
        # All input eventually becomes legendary (minus losses from recycling)
        flows, _, _ = recycler_loop(100.0, quality_chance=0.25, quality_to_keep=5)
        # Legendary output should be less than input (due to 75% loss per recycle)
        assert flows[4] < 100.0
        assert flows[4] > 0

    def test_higher_quality_chance_more_legendary(self):
        """Higher quality chance should produce more legendary items."""
        flows_low, _, _ = recycler_loop(100.0, quality_chance=0.1)
        flows_high, _, _ = recycler_loop(100.0, quality_chance=0.5)
        assert flows_high[4] > flows_low[4]


class TestAssemblerRecyclerParameters:
    """Tests for parameter generation functions."""

    def test_assembler_params_length(self):
        """Should return NUM_TIERS parameters."""
        # get_assembler_parameters expects a list of tuples (one per tier)
        params = get_assembler_parameters([(0, 4)] * NUM_TIERS)
        assert len(params) == NUM_TIERS

    def test_assembler_params_last_tier_zero(self):
        """Last tier (legendary) should have zero quality chance."""
        params = get_assembler_parameters([(0, 4)] * NUM_TIERS, quality_to_keep=5)
        assert params[4] == (0, 0)

    def test_recycler_params_length(self):
        """Should return NUM_TIERS parameters."""
        params = get_recycler_parameters()
        assert len(params) == NUM_TIERS

    def test_recycler_params_kept_tiers_zero(self):
        """Kept tiers should have zero parameters."""
        params = get_recycler_parameters(quality_to_keep=5)
        assert params[4] == (0, 0)


class TestTransitionMatrix:
    """Tests for create_transition_matrix function."""

    def test_matrix_shape(self):
        """Matrix should be 2*NUM_TIERS x 2*NUM_TIERS."""
        assembler = np.eye(NUM_TIERS)
        recycler = np.eye(NUM_TIERS)
        mat = create_transition_matrix(assembler, recycler)
        assert mat.shape == (NUM_TIERS * 2, NUM_TIERS * 2)

    def test_assembler_in_upper_right(self):
        """Assembler matrix should be in upper right quadrant."""
        assembler = np.ones((NUM_TIERS, NUM_TIERS)) * 2
        recycler = np.ones((NUM_TIERS, NUM_TIERS)) * 3
        mat = create_transition_matrix(assembler, recycler)

        # Upper right quadrant (ingredients -> products)
        for i in range(NUM_TIERS):
            for j in range(NUM_TIERS):
                assert mat[i, j + NUM_TIERS] == 2

    def test_recycler_in_lower_left(self):
        """Recycler matrix should be in lower left quadrant."""
        assembler = np.ones((NUM_TIERS, NUM_TIERS)) * 2
        recycler = np.ones((NUM_TIERS, NUM_TIERS)) * 3
        mat = create_transition_matrix(assembler, recycler)

        # Lower left quadrant (products -> ingredients)
        for i in range(NUM_TIERS):
            for j in range(NUM_TIERS):
                assert mat[i + NUM_TIERS, j] == 3


class TestAssemblerRecyclerLoop:
    """Tests for assembler_recycler_loop function."""

    def test_output_shape(self):
        """Output flow vector should have 2*NUM_TIERS elements."""
        flows, _, _, _ = assembler_recycler_loop(
            100.0,
            assembler_modules_config=(0, 4),
        )
        assert flows.shape == (NUM_TIERS * 2,)

    def test_scalar_input(self):
        """Scalar input should be converted to array."""
        flows, _, _, _ = assembler_recycler_loop(
            100.0,
            assembler_modules_config=(0, 4),
        )
        assert isinstance(flows, np.ndarray)

    def test_legendary_ingredients_output(self):
        """Should produce legendary ingredients when ingredient_quality_to_keep=5."""
        # Use full quality config with base productivity to ensure output
        flows, _, _, _ = assembler_recycler_loop(
            100.0,
            assembler_modules_config=(0, 4),
            ingredient_quality_to_keep=5,
            product_quality_to_keep=None,
            base_prod_bonus=1.0,  # Need productivity to produce items
        )
        # Index 4 is legendary ingredients
        assert flows[4] > 0

    def test_legendary_products_output(self):
        """Should produce legendary products when product_quality_to_keep=5."""
        # Use full quality config with base productivity to ensure output
        flows, _, _, _ = assembler_recycler_loop(
            100.0,
            assembler_modules_config=(0, 4),
            ingredient_quality_to_keep=None,
            product_quality_to_keep=5,
            base_prod_bonus=1.0,  # Need productivity to produce items
        )
        # Index 9 is legendary products (NUM_TIERS + 4)
        assert flows[9] > 0

    def test_returns_transition_matrix(self):
        """Should return a valid transition matrix."""
        _, mat, _, _ = assembler_recycler_loop(
            100.0,
            assembler_modules_config=(0, 4),
        )
        assert mat.shape == (NUM_TIERS * 2, NUM_TIERS * 2)

    def test_list_config_per_tier(self):
        """Should accept list of configs per tier."""
        config = [(0, 4), (0, 4), (0, 4), (0, 4), (4, 0)]
        flows, _, _, _ = assembler_recycler_loop(
            100.0,
            assembler_modules_config=config,
        )
        assert flows.shape == (NUM_TIERS * 2,)

    
