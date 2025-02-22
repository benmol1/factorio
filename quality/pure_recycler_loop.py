from quality import create_production_matrix
import numpy as np
from functools import lru_cache
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd

NUM_TIERS = 5

@lru_cache()
def recycler_matrix(quality_chance: float, quality_to_keep: int = 5, production_ratio: float = 0.25) -> np.ndarray:
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
    assert production_ratio >= 0

    recycling_rows = quality_to_keep - 1
    saving_rows = 5 - recycling_rows

    return create_production_matrix([(quality_chance, production_ratio)] * recycling_rows + [(0, 0)] * saving_rows)


def recycler_loop(
    input_vector: Union[np.array, float],
    quality_chance: float,
    quality_to_keep: int = 5,
    production_ratio: float = 0.25,
    speed_recycler: float = 0.4,
    num_recyclers: int = 1,
    recipe_time: float = 1,
    recipe_ratio: float = 1,
    verbose: bool = False,
) -> np.ndarray:
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

    transition_matrix = recycler_matrix(quality_chance, quality_to_keep, production_ratio)

    crafting_time_vector = create_crafting_time_vector(
        speed_recycler, num_recyclers, recipe_time
    )

    if verbose:
        print("## Transition matrix:\n", transition_matrix)
        print("## Crafting time vector:\n", crafting_time_vector)

    # Initialise loop variable and output arrays
    ii = 0
    result_flows = [input_vector]
    crafting_time = [[0, 0, False]]

    while True:
        ii += 1
        ct_this_iteration = compute_crafting_time(
            result_flows[-1], crafting_time[-1], recipe_ratio, crafting_time_vector
        )

        crafting_time.append(ct_this_iteration)
        result_flows.append(result_flows[-1] @ transition_matrix)

        if sum(abs(result_flows[-2] - result_flows[-1])) < 1e-2:
            # There's nothing left in the system
            break

    # Create the output dataframe
    col_headers = ["P1", "P2", "P3", "P4", "P5"]
    output_df = pd.DataFrame(data=result_flows, columns=col_headers)
    crafting_time_df = pd.DataFrame(data=crafting_time, columns=["Crafting time", "Max time index", "Bottleneck"])
    output_df = output_df.join(crafting_time_df)

    if verbose:
        print("## Iterations:")
        print(output_df)

        if sum(output_df["Bottleneck"] > 0):
            print(output_df[output_df["Bottleneck"] == True])
            output_df.to_csv("output_df.csv")

    return sum(result_flows)


def create_crafting_time_vector(speed_recycler: float = 0.4,  # the speed of a normal recycler with 4x quality modules
                                num_recyclers: int = 1,
                                recipe_time: float = 1) -> np.ndarray:

    res = [recipe_time / (16 * speed_recycler)] * NUM_TIERS
    res = np.array(res) / num_recyclers

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


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=1000)
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.max_rows", 20)
    pd.set_option("colheader_justify", "right")
    pd.options.display.float_format = "{:.1f}".format

    q = 4 * 0.062

    # recycler loop for biter eggs
    print(recycler_loop(input_vector=16,
                        quality_chance=q,
                        recipe_time=2,
                        num_recyclers=4,
                        speed_recycler=1,  # legendary recyclers
                        verbose=True))

    efficiency_output = 1 / recycler_loop(1, q, verbose=False)[4]

    print(efficiency_output)

    # # Define two ranges for x and y
    # x_values = quality_range
    # y_values = efficiency_output

    # # Create scatter plot
    # plt.plot(x_values, y_values, color="blue")
    # plt.yscale("log")

    # # Labels and title
    # plt.xlabel("Quality chance")
    # plt.ylabel("Efficiency")
    # plt.title("Efficiency of crusher loop, by quality chance")
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # # Show legend
    # plt.legend()

    # # Display plot
    # plt.show()
