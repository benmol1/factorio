from quality import create_production_matrix
import numpy as np
from functools import lru_cache
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd

NUM_TIERS = 5


@lru_cache()
def recycler_matrix(quality_chance: float, quality_to_keep: int = 5, is_asteroid_crusher: bool = False) -> np.ndarray:
    """Returns a matrix of a recycler with quality chance `quality_chance`
    that saves any item of quality level `quality_to_keep` or above.

    Args:
        quality_chance (float): Quality chance of the recyclers (in %).
        quality_to_keep (int): Minimum quality level of the items to be removed from the system
            (By default only removes legendaries).
        production_ratio (float): Productivity ratio of the recyclers (0.25 by default)

    Returns:
        np.ndarray: Standard production matrix.
    """
    assert quality_chance > 0
    assert type(quality_to_keep) == int and 1 <= quality_to_keep <= 5

    # Set the production ratio for this loop
    if is_asteroid_crusher:
        production_ratio = 0.8
    else:
        production_ratio = 0.25

    recycling_rows = quality_to_keep - 1
    saving_rows = 5 - recycling_rows

    return create_production_matrix([(quality_chance, production_ratio)] * recycling_rows + [(0, 0)] * saving_rows)


def recycler_loop(
    input_vector: Union[np.array, float],
    quality_chance: float,
    quality_to_keep: int = 5,
    speed_recycler: float = 0.4,
    num_recyclers: Union[np.array, int] = 1,
    recipe_time: float = 1.0,
    recipe_ratio: float = 1.0,
    is_asteroid_crusher: bool = False,
    verbose: bool = False,
) -> (np.ndarray, np.ndarray):
    """Returns a vector with values for each quality level that mean different things,
    depending on whether that quality is kept or recycled:
        - If the quality is kept: the value is the production rate of items of that quality level.
        - If the quality is recycled: the value is the internal flow rate of items of that quality level in the system.

    Args:
        input_vector (np.array): The flow rate of items going into the system. If a single value is passed,
            it is assumed to be the input rate of Q1 items going into the system.
        quality_chance (float): Quality chance of the recycler loop (in %).
        quality_to_keep (int): Minimum quality level of the items to be removed from the system
            (By default only removes legendaries).

    Returns:
        np.ndarray: Vector with values for each quality level.
    """
    if type(input_vector) in (float, int):
        input_vector = np.array([input_vector, 0, 0, 0, 0])

    if type(num_recyclers) == int:
        num_recyclers = np.array([num_recyclers] * NUM_TIERS)

    transition_matrix = recycler_matrix(quality_chance, quality_to_keep, is_asteroid_crusher)
    crafting_time_vector = create_crafting_time_vector(speed_recycler, num_recyclers, recipe_time, is_asteroid_crusher)

    if verbose:
        print("## Transition matrix:\n", transition_matrix)
        print("## Crafting time vector:\n", crafting_time_vector)

    # Initialise loop variable and output arrays
    ii = 0
    result_flows = [input_vector]
    crafting_time = [[0.0, 0, False]]
    total_crafting_time = 0

    while True:
        ii += 1
        ct_this_iteration = compute_crafting_time(
            result_flows[-1], crafting_time[-1], recipe_ratio, crafting_time_vector
        )

        total_crafting_time += ct_this_iteration[0]
        crafting_time.append(ct_this_iteration)
        result_flows.append(result_flows[-1] @ transition_matrix)

        if sum(abs(result_flows[-2] - result_flows[-1])) < 1e-6:
            # There's nothing left in the system
            break

    # Create the output dataframe
    col_headers = ["P1", "P2", "P3", "P4", "P5"]
    output_df = pd.DataFrame(data=result_flows, columns=col_headers)
    crafting_time_df = pd.DataFrame(data=crafting_time, columns=["Crafting time", "Max time index", "Bottleneck"])
    output_df = output_df.join(crafting_time_df)

    if verbose:
        print("## Iterations:")
        print(output_df.head(10))

        if sum(output_df["Bottleneck"] > 0):
            print(output_df[output_df["Bottleneck"] == True])
            output_df.to_csv("output_df.csv")

    return sum(result_flows), transition_matrix, total_crafting_time


def create_crafting_time_vector(
    speed_recycler: float = 0.4,  # the speed of a normal recycler with 4x quality modules
    num_recyclers: np.ndarray = np.array([1.0] * NUM_TIERS),
    recipe_time: float = 1,
    is_asteroid_crusher: bool = False,
) -> np.ndarray:

    res = [recipe_time / (16 * speed_recycler)] * NUM_TIERS
    res = np.array(res) / num_recyclers

    # If this is an asteroid crusher, re-multiply the crafting time by 16 because that discount
    # only applies to recycling
    if is_asteroid_crusher:
        res *= 16

    return np.array(res)


def compute_crafting_time(
    input_flows: np.ndarray, prev_crafting_time: list, recipe_ratio: float, ct_vector: np.ndarray
) -> list:

    ct = 0

    # The recyclers are in series, so the total crafting time is relevant
    recycler_crafting_time = np.dot(input_flows, ct_vector)
    ct += recycler_crafting_time

    # This is a pure recycler loop, so the recyclers must always be the most time-consuming component of each iteration
    max_time_index = NUM_TIERS + 1

    # Compare the crafting time of this iteration to the previous one; if the crafting time here is greater then
    # report the bottleneck
    bottleneck = False
    if (prev_crafting_time[0] > 0) and (ct > 1) and (ct - prev_crafting_time[0] > 0.1):
        bottleneck = True

    return [ct, max_time_index, bottleneck]


def normal_to_legendary_ratio():
    indices = list(range(1, 25)) + [24.8]
    ratios = [float(1 / recycler_loop(1, i)[4]) for i in indices]

    print(f"{indices[9:]=}")
    print(f"{ratios[9:]=}")


def efficiency_data():
    indices = list(range(1, 25)) + [24.8]

    uncommon = [float(recycler_loop(100, i, 2)[1]) for i in indices]
    rare = [float(recycler_loop(100, i, 3)[2]) for i in indices]
    epic = [float(recycler_loop(100, i, 4)[3]) for i in indices]
    legendary = [float(recycler_loop(100, i, 5)[4]) for i in indices]

    print(f"{uncommon=}")
    print(f"{rare=}")
    print(f"{epic=}")
    print(f"{legendary=}")


def get_production_rate(
    input_vector: np.ndarray, output_flows: np.ndarray, transition_matrix: np.ndarray, is_asteroid_crusher: bool = False
) -> np.ndarray:
    """
    Computes the production rate at each quality level.

    The game's internal production tracker only captures item-created and item-destroyed events. Therefore when an item
    is recycled into itself this doesn't show as either. This function specifically computes the rate at which items are
    produced, as opposed to the total flows - this is directly comparable with the game's production statistics panel

    """

    # in order to provide the input_vector, it must first be produced somewhere
    production_rate = input_vector

    # For typical recycling loops, the only way to produce an item is to upcycle its equivalents from the tiers below.
    # These up-cycling flows are given by multiplying total outputs flows per tier by the relevant cell(s) of the
    # transition matrix.
    # For asteroid crushers this is not true, as equal-tier production events happen frequently.
    # Ignoring quality and the 0.8 productivity factor, 50% of asteroid reprocessing crafts return
    # an equiv-tier product. So whilst we typically would ignore equiv-tier (=on-diagonal)
    # transitions, for the asteroid crusher we apply a scalar of 0.5)
    for ii in range(0, NUM_TIERS):
        prod_rate = 0
        for jj in range(ii + 1):
            if jj == ii:
                scalar = 0
                if is_asteroid_crusher:
                    scalar = 0.5
            else:
                scalar = 1
            prod_rate += output_flows[jj] * transition_matrix[jj][ii] * scalar
        production_rate[ii] += prod_rate

    return production_rate


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=1000)
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.max_rows", 20)
    pd.set_option("colheader_justify", "right")
    pd.options.display.float_format = "{:.2f}".format

    # recycler loop for biter eggs
    # input_vector = np.array([32.0, 0.0, 0.0, 0.0, 0.0])
    # q = 4 * 0.062
    # results = recycler_loop(
    #     input_vector=input_vector,
    #     quality_chance=q,
    #     recipe_time=10,
    #     num_recyclers=28,
    #     speed_recycler=1,  # legendary recyclers
    #     verbose=True,
    # )

    # recycler loop for asteroids eggs
    input_vector = np.array([8.33, 0.0, 0.0, 0.0, 0.0])
    q = 2 * 0.062
    results = recycler_loop(
        input_vector=input_vector,
        quality_chance=q,
        recipe_time=2,
        num_recyclers=np.array([24, 8, 3, 1, 1]),
        speed_recycler=2.25,  # legendary crushers, each with 2x qual modules
        is_asteroid_crusher=True,
        verbose=True,
    )

    flows = results[0]
    transition_matrix = results[1]
    total_crafting_time = results[2]

    print("## Flow per second:")
    print(flows)

    print("## Flow per minute, per type:")
    print(flows * 60)

    production_rate = get_production_rate(input_vector, flows, transition_matrix, is_asteroid_crusher=True)

    print("## Production rates per minute, per type:")
    print(production_rate * 60)
    print("## Legendary production rate per hour: %.1f" % (production_rate[4] * 3600))

    print("## Total crafting time: %.2f seconds" % total_crafting_time)

    print("## Suggested ratio of asteroid crushers at each tier:")
    print(flows / flows[3] * 3)
