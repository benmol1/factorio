from enum import IntEnum

import numpy as np

NUM_TIERS = 5
BEST_PROD_MODULE = 0.250  # [0.100, 0.130, 0.160, 0.190, 0.250]
BEST_QUAL_MODULE = 0.062  # [0.025, 0.032, 0.040, 0.047, 0.062]


class QualityTier(IntEnum):
    """Enumeration of quality tiers in Factorio's quality system."""

    Normal = 0
    Uncommon = 1
    Rare = 2
    Epic = 3
    Legendary = 4


def quality_probability(quality_chance: float, input_tier: QualityTier, output_tier: QualityTier) -> float:
    """Calculate the probability of upgrading from one quality tier to another.

    Calculates the probability of a machine craft with a certain `quality_chance` upgrading
    the resulting product from the tier of the products (`input_tier`) to the `output_tier`.

    Args:
        quality_chance: Quality chance as a float between 0 and 1.
        input_tier: Quality tier of the ingredients.
        output_tier: Quality tier of the product.

    Returns:
        A probability from 0 to 1.
    """
    # Basic validations
    assert 0 <= quality_chance <= 1
    assert 0 <= input_tier <= (NUM_TIERS - 1) and isinstance(input_tier, int)
    assert 0 <= output_tier <= (NUM_TIERS - 1) and isinstance(output_tier, int)

    # Some QoL conversions
    i = input_tier
    o = output_tier

    # An item can never be downgraded
    if input_tier > output_tier:
        return 0

    # If the item is already in the top quality tier, it will remain so
    if input_tier == NUM_TIERS - 1:
        return 1

    # Probability of item staying in the same tier
    if input_tier == output_tier:
        return 1 - quality_chance

    # Probability of item going straight to the top quality tier
    if output_tier == NUM_TIERS - 1:
        return quality_chance / (10 ** ((NUM_TIERS - 2) - i))

    # else
    return (quality_chance * 9 / 10) / (10 ** (o - i - 1))


def quality_matrix(quality_chance: float) -> np.ndarray:
    """Return the quality transition matrix for a given quality chance.

    The matrix indicates the probabilities of any input tier jumping to any other tier.

    Args:
        quality_chance: Quality chance as a decimal (e.g., 0.25 for 25%).

    Returns:
        NxN matrix where rows are input quality and columns are output quality.
    """
    res = np.zeros((NUM_TIERS, NUM_TIERS))

    for row in range(NUM_TIERS):
        for column in range(NUM_TIERS):
            res[row][column] = quality_probability(quality_chance, row, column)

    return res


def create_production_matrix(parameters_per_row: list[tuple[float, float]]) -> np.ndarray:
    """Create a production matrix with per-row quality chance and production ratio.

    Args:
        parameters_per_row: List of five tuples. Each tuple contains
            (quality_chance, production_ratio) for the respective row.

    Returns:
        NxN production matrix combining quality transitions with production ratios.
    """
    # Basic validations
    assert len(parameters_per_row) == NUM_TIERS
    assert isinstance(parameters_per_row, list)
    for pair in parameters_per_row:
        assert isinstance(pair, tuple)
        assert len(pair) == 2

    res = np.zeros((NUM_TIERS, NUM_TIERS))

    for row in range(NUM_TIERS):
        quality_chance, production_ratio = parameters_per_row[row]

        for column in range(NUM_TIERS):
            res[row][column] = quality_probability(quality_chance, row, column) * production_ratio

    return res


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    qual_mat_EMP = quality_matrix(0.235)

    print(qual_mat_EMP)

    params_BC_em_plant = [(0.235, 1.5)] * (NUM_TIERS - 1) + [(0, 1.5)]
    prod_mat_BC_em_plant = create_production_matrix(params_BC_em_plant)

    print(prod_mat_BC_em_plant)
