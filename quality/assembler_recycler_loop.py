import numpy as np
from typing import Union, List, Tuple
import itertools
from tqdm import tqdm
from enum import Enum
import pandas

from quality import create_production_matrix

NUM_TIERS = 3
BEST_PROD_MODULE = 0.100 # [0.100, 0.130, 0.160, 0.190, 0.250]
BEST_QUAL_MODULE = 0.040 # [0.025, 0.032, 0.040, 0.047, 0.062]

def create_transition_matrix(assembler_matrix : np.ndarray, recycler_matrix : np.ndarray) -> np.ndarray:
    """Creates a transition matrix based on the 
    provided recycler and assembler production matrices.

    Args:
        assembler_matrix (np.ndarray): Assembler production matrix.
        recycler_matrix (np.ndarray): Recycler production matrix.

    Returns:
        np.ndarray: Transition matrix with the recycler production matrix 
        in the lower left and assembler production matrix in the upper right.
    """
    res = np.zeros((NUM_TIERS*2,NUM_TIERS*2))

    for i in range(NUM_TIERS):
        for j in range(NUM_TIERS):
            res[i + NUM_TIERS][j] = recycler_matrix[i, j]
            res[i][j + NUM_TIERS] = assembler_matrix[i, j]

    return res


def create_crafting_time_vector(speed_assembler : float, speed_recycler : float, recipe_time : float) -> np.ndarray:

    res = [recipe_time / speed_assembler] * NUM_TIERS + [recipe_time / (16 * speed_recycler)] * NUM_TIERS

    return np.array(res)


def create_crafting_time_matrix(transition_matrix : np.ndarray, assembler_speed : float, recycler_speed : float) -> np.ndarray:

    res = np.zeros((NUM_TIERS*2,NUM_TIERS*2))

    for i in range(NUM_TIERS*2):
        for j in range(NUM_TIERS*2):
            if i < NUM_TIERS:
                res[i][j] = transition_matrix[i][j] / assembler_speed
            else:
                res[i][j] = transition_matrix[i][j] / (recycler_speed) / 16

    return res

def get_assembler_parameters(
        assembler_modules_config: Union[Tuple[float, float], List[Tuple[float, float]]],
        # Modules configuration of assemblers for every quality level
        quality_to_keep: int = NUM_TIERS,  # Don't assemble legendary ingredients (default)
        base_prod_bonus: float = 0,  # base productivity of assembler + productivity technologies
        recipe_ratio: float = 1,  # Ratio of items to ingredients of the recipe
        prod_module_bonus: float = BEST_PROD_MODULE,
        qual_module_bonus: float = BEST_QUAL_MODULE) -> List[Tuple[float, float]]:
    production_rows = quality_to_keep - 1

    res = [(0, 0)] * NUM_TIERS

    for i, (nP, nQ) in enumerate(assembler_modules_config):
        if i == production_rows:
            break

        # Assembler stats
        production_ratio = min(base_prod_bonus + nP * prod_module_bonus, 4) * recipe_ratio
        quality_chance = nQ * qual_module_bonus

        res[i] = (quality_chance, production_ratio)

    return res

def get_recycler_parameters(
        quality_to_keep : int = NUM_TIERS, # Don't recycle max_quality products (default)
        recipe_ratio : float = 1, # Ratio of items to ingredients of the recipe
        qual_module_bonus : float = BEST_QUAL_MODULE) -> List[Tuple[float, float]]:
    
    recycling_rows = quality_to_keep - 1
    saving_rows = NUM_TIERS - recycling_rows

    # Recycler stats
    production_ratio = 0.25 / recipe_ratio
    quality_chance = 4 * qual_module_bonus

    return [(quality_chance, production_ratio)] * recycling_rows + [(0, 0)] * saving_rows


def assembler_recycler_loop(
        input_vector : Union[np.array, float],
        assembler_modules_config : Union[Tuple[float, float], List[Tuple[float, float]]], # Modules configuration of assemblers for every quality level
        product_quality_to_keep : Union[int, None] = NUM_TIERS, # Don't recycle top tier products (default)
        ingredient_quality_to_keep : Union[int, None] = NUM_TIERS, # Don't assemble top tier ingredients (default)
        base_prod_bonus : float = 0, # base productivity of assembler + productivity technologies
        recipe_ratio : float = 1, # Ratio of items to ingredients of the recipe
        prod_module_bonus : float = BEST_PROD_MODULE,
        qual_module_bonus : float = BEST_QUAL_MODULE,
        speed_assembler : float = 2,
        speed_recycler : float = 0.5,
        recipe_time : float = 1,
        print_crafting_time_matrix : float = False
) -> np.array:
    """Returns a vector with values for each quality level that mean different things, depending on whether that quality is kept or recycled:
        - If the quality is kept: the value is the production rate of ingredients/items of that quality level.
        - If the quality is recycled: the value is the internal flow rate of ingredients/items of that quality level in the system.
    
    The first five values represent the ingredients and the last five values represent the items.

    Args:
        input_vector (Union[np.array, float]): The ingredients and items intake of the system.
        assembler_modules_config (Union[Tuple[float, float], List[Tuple[float, float]]]): Number of productivity and quality modules for the assemblers of each quality of item.
        product_quality_to_keep (Union[int, None], optional): Minimum quality level of the items to be removed from the system.
        ingredient_quality_to_keep (Union[int, None], optional): Minimum quality level of the ingredients to be removed from the system.
        base_prod_bonus (float, optional): Base productivity of assembler + productivity technologies. Defaults to 0.
        recipe_ratio (float, optional): Ratio of items to ingredients of the crafting recipe. Defaults to 1.
        prod_module_bonus (float, optional): Productivity bonus from productivity modules.
        qual_module_bonus (float, optional): Quality chance bonus from quality modules.
        speed_assembler
        speed_recycler

    Returns:
        np.array: Vector with values for each quality level. The first five values represent the ingredients and the last five values represent the items.
    """
    
    if type(assembler_modules_config) == tuple:
        assembler_modules_config = [assembler_modules_config] * NUM_TIERS
    elif type(assembler_modules_config) == list:
        assert len(assembler_modules_config) == NUM_TIERS

    # Parameters for the production matrices
    assembler_parameters = get_assembler_parameters(
        assembler_modules_config,
        ingredient_quality_to_keep if ingredient_quality_to_keep != None else NUM_TIERS + 1,
        base_prod_bonus,
        recipe_ratio,
        prod_module_bonus,
        qual_module_bonus
    )
    recycler_parameters  = get_recycler_parameters(
        product_quality_to_keep if product_quality_to_keep != None else NUM_TIERS + 1,
        recipe_ratio,
        qual_module_bonus
    )
    # Create the transition matrix
    transition_matrix = create_transition_matrix(
        assembler_matrix=create_production_matrix(assembler_parameters),
        recycler_matrix=create_production_matrix(recycler_parameters)
    )

    crafting_time_vector = create_crafting_time_vector(
        speed_assembler,
        speed_recycler,
        recipe_time
    )

    if print_crafting_time_matrix:
        print("Transition matrix:\n", transition_matrix)
        print("Crafting time vector:\n", crafting_time_vector)

    # Handle the case where input_vector has just been given as a single scalar value
    if type(input_vector) in (float, int):
        input_vector = np.array([input_vector] + [0] * (NUM_TIERS*2 - 1))
    
    result_flows = [input_vector]
    while True:
        result_flows.append(result_flows[-1] @ transition_matrix)

        if sum(abs(result_flows[-2] - result_flows[-1])) < 1E-2:
            # There's nothing left in the system
            break

    return sum(result_flows)

def get_config_string(config : List[Tuple[int, int]]):
    return [f"{p}P{q}Q" for (p, q) in config]

class SystemOutput(Enum):
    INGREDIENTS = 0
    ITEMS = 1

class ModuleStrategy(Enum):
    FULL_QUALITY = 0
    FULL_PRODUCTIVITY = 1
    OPTIMIZE = 2

def get_all_configs(module_slots : int):
    "Generate all possible configurations for an assembler with `n` module slots."
    module_variations_for_assembler = []

    for p in range(module_slots + 1):
        q = module_slots - p
        module_variations_for_assembler.append((p, q))
    
    res = list(itertools.product(* [module_variations_for_assembler] * NUM_TIERS))

    for i in range(len(res)):
        res[i] = list(res[i])
    
    return res

def assembler_recycler_efficiency(
        module_slots : int,
        base_productivity : float,
        system_output : SystemOutput,
        module_strategy : ModuleStrategy,
        prod_mod_bonus : float = BEST_PROD_MODULE,
        qual_mod_bonus : float = BEST_QUAL_MODULE) -> float:
    "Returns the efficiency of the setup with the given parameters (%)."
    assert module_slots >= 0 and base_productivity >= 0

    if system_output == SystemOutput.ITEMS:
        keep_items = NUM_TIERS
        keep_ingredients = None
    else: # system_output == SystemOutput.INGREDIENTS:
        keep_items = None
        keep_ingredients = NUM_TIERS
    
    # What is the output of the system: ingredients or items?
    result_index = (NUM_TIERS - 1) if system_output == SystemOutput.INGREDIENTS else (NUM_TIERS * 2 - 1)
    
    if module_strategy != ModuleStrategy.OPTIMIZE:
        if module_strategy == ModuleStrategy.FULL_PRODUCTIVITY:
            config = (module_slots, 0)
        else:
            config = (0, module_slots)
        
        output = assembler_recycler_loop(100, config, keep_items, keep_ingredients, base_productivity,
                                         prod_module_bonus=prod_mod_bonus, qual_module_bonus=qual_mod_bonus)
        return output[result_index]
    else:
        best_config = None
        best_efficiency = 0

        all_configs = get_all_configs(module_slots)

        for config in tqdm(list(all_configs)):
            if config[NUM_TIERS-1] != (module_slots, 0):
                # Makes no sense to put quality modules on legendary item crafter
                continue

            output = assembler_recycler_loop(100, config, keep_items, keep_ingredients, base_productivity,
                                             prod_module_bonus=prod_mod_bonus, qual_module_bonus=qual_mod_bonus)
            efficiency = float(output[result_index])

            if best_efficiency < efficiency:
                best_config = config
                best_efficiency = efficiency
                print(config, ": %.2f" % efficiency)
        
        return best_efficiency

def efficiency_table():
    DATA = { # (number of slots, base productivity)
        "Electric furnace/Centrifuge" : (2, 0),
        "Chemical Plant"              : (3, 0),
        "Assembling machine"          : (4, 0),
        "Foundry/Biochamber"          : (4, 0.5),
        "Electromagnetic plant"       : (5, 0.5),
        "Cryogenic plant"             : (8, 0),
    }
    OUTPUTS = (SystemOutput.ITEMS, SystemOutput.INGREDIENTS)
    STRATEGIES = (
        ModuleStrategy.FULL_QUALITY,
        ModuleStrategy.FULL_PRODUCTIVITY,
        ModuleStrategy.OPTIMIZE
    )
    KEY_NAMES = {
        (SystemOutput.ITEMS, ModuleStrategy.FULL_QUALITY) : "(D) Quality only, max items",
        (SystemOutput.ITEMS, ModuleStrategy.FULL_PRODUCTIVITY) : "(E) Prod only, max items",
        (SystemOutput.ITEMS, ModuleStrategy.OPTIMIZE) : "(F) Optimal modules, max items",
        (SystemOutput.INGREDIENTS, ModuleStrategy.FULL_QUALITY) : "(G) Quality only, max ingredients",
        (SystemOutput.INGREDIENTS, ModuleStrategy.FULL_PRODUCTIVITY) : "(H) Prod only, max ingredients",
        (SystemOutput.INGREDIENTS, ModuleStrategy.OPTIMIZE) : "(I) Optimal modules, max ingredients",
    }

    table = {key : {} for key in DATA}

    for assembler_type, (slots, base_prod) in DATA.items():
        for output in OUTPUTS:
            for strategy in STRATEGIES:
                eff = assembler_recycler_efficiency(slots, base_prod, output, strategy)
                table[assembler_type][KEY_NAMES[(output, strategy)]] = eff
    
    print(pandas.DataFrame(table).T.to_string())


def verbose_AR_loop(qual_assembler, prod_assembler,
                    qual_recycler, prod_recycler=0.25,
                    speed_assembler=1, speed_recycler=0.5, recipe_time=1):

    # Create and print the transition matrix
    recycler_matrix = create_production_matrix([(qual_recycler, prod_recycler)] * (NUM_TIERS - 1) + [(0, 0)])
    em_plant_matrix = create_production_matrix([(qual_assembler, prod_assembler)] * (NUM_TIERS - 1) + [(0, prod_assembler)])

    transition_matrix = create_transition_matrix(
        assembler_matrix=em_plant_matrix,
        recycler_matrix=recycler_matrix
    )

    crafting_time_vector = create_crafting_time_vector(speed_assembler, speed_recycler, recipe_time)

    print("## Transition matrix:\n", transition_matrix, "\n")
    print("## Crafting time vector:\n", crafting_time_vector, "\n")
    # Create and print the input vector
    input_vector = np.array([1] + [0] * (NUM_TIERS * 2 - 1))
    ii = 0
    print("## Iterations:")
    print(ii, "\t", input_vector)
    result_flows = [input_vector]
    # Apply the transition matrix iteratively, until the resulting flows become very small
    while True:
        ii += 1

        # Compute & print the flows for this iteration
        result_flows_this_iteration = result_flows[-1] @ transition_matrix
        print(ii, "\t", result_flows_this_iteration)

        # append this iteration to the overall results array
        result_flows.append(result_flows_this_iteration)

        if sum(abs(result_flows[-2] - result_flows[-1])) < 1E-2:
            # There's nothing left in the system
            break
    # Print the sum of all flows.
    # In the simple case, the output is the number of highest-tier products resulting from the input vector
    print(sum(result_flows))


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True, linewidth = 1000)

    n_slots = 5
    base_prod = 1.5

    verbose_AR_loop(qual_assembler=n_slots*BEST_QUAL_MODULE,
                    prod_assembler=base_prod,
                    qual_recycler=4*BEST_QUAL_MODULE,
                    speed_assembler=2,
                    speed_recycler=0.5,
                    recipe_time=60)

    # # Compact AR loop for an EM plants at our current tech level, with the classic [Q^(n-1)P] strategy
    # output_flows = assembler_recycler_loop(input_vector=100,
    #                                        assembler_modules_config=[(0, n_slots)] * (NUM_TIERS-1) + [(n_slots, 0)],
    #                                        product_quality_to_keep=NUM_TIERS,
    #                                        ingredient_quality_to_keep=None,
    #                                        base_prod_bonus=base_prod, recipe_ratio=1,
    #                                        prod_module_bonus=0,
    #                                        qual_module_bonus=BEST_QUAL_MODULE,
    #                                        assembler_speed=2,
    #                                        recycler_speed=0.5,
    #                                        recipe_time=60,
    #                                        print_crafting_time_matrix=True)
    # print(output_flows, "\n")
    #
    #
    # output = SystemOutput.ITEMS
    # strategy = ModuleStrategy.OPTIMIZE
    #
    # eff = assembler_recycler_efficiency(n_slots, base_prod, output, strategy,
    #                                     prod_mod_bonus=BEST_PROD_MODULE,
    #                                     qual_mod_bonus=BEST_QUAL_MODULE)

