import numpy as np

from enum import IntEnum
from typing import Union, List, Tuple

NUM_TIERS = 4


class QualityTier(IntEnum):
    Normal = 0
    Uncommon = 1
    Rare = 2
    Epic = 3
    Legendary = 4


def quality_probability(quality_chance: float, input_tier: QualityTier, output_tier: QualityTier) -> float:
    """Calculates the probability of a machine craft with a certain `quality_chance` upgrading
    the resulting product from the tier of the products (`input_tier`) to the `output_tier`.

    Args:
        quality_chance (float): Quality chance
        input_tier (QualityTier): Quality tier of the ingredients.
        output_tier (QualityTier): Quality tier of the product.

    Returns:
        float: A probability from 0 to 1.
    """

    # Basic validations
    assert 0 <= quality_chance <= 1
    assert 0 <= input_tier <= (NUM_TIERS - 1) and type(input_tier) == int
    assert 0 <= output_tier <= (NUM_TIERS - 1) and type(output_tier) == int

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
    """Returns the quality matrix for the corresponding `quality_chance` which indicates
    the probabilities of any input tier jumping to any other tier.

    Args:
        quality_chance (float): Quality chance (in %).

    Returns:
        np.ndarray: nxn matrix. The input quality is split by row; output quality by column
    """

    res = np.zeros((NUM_TIERS, NUM_TIERS))

    for row in range(NUM_TIERS):
        for column in range(NUM_TIERS):
            res[row][column] = quality_probability(quality_chance, row, column)

    return res


def basic_production_matrix(quality_chance: float, production_ratio: float = 1) -> np.ndarray:
    "Returns the production matrix for the corresponding `quality_chance` and `production_ratio`."
    return quality_matrix(quality_chance) * production_ratio


def create_production_matrix(parameters_per_row: List[Tuple[float, float]]) -> np.ndarray:
    """Returns a production matrix where every row has a specific quality chance and prodution ratio.

    Args:
        parameters_per_row (List[Tuple[float, float]]): List of five tuples. Each tuple indicates the
            quality chance (%) and production ratio for the respective row.

    Returns:
        np.ndarray: nxn production matrix.
    """

    # Basic validations
    assert len(parameters_per_row) == NUM_TIERS
    assert type(parameters_per_row) == list
    for pair in parameters_per_row:
        assert type(pair) == tuple
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
