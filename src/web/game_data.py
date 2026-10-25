"""Factorio: Space Age machine and module stats, used to turn game terms into simulation parameters."""

from dataclasses import dataclass

from src.quality.utils import NUM_TIERS
from src.web.recipes import recipe_data

# Machine speed multiplier per machine quality (Normal → Legendary)
QUALITY_SPEED_MULTIPLIER = [1.0, 1.3, 1.6, 1.9, 2.5]

# Speed penalty per module, which doesn't change with module quality
QUALITY_MODULE_SPEED_PENALTY = 0.05
PROD_MODULE_SPEED_PENALTY = {1: 0.05, 2: 0.10, 3: 0.15}

# Machines can't be slowed below 20% of their base speed
MIN_SPEED_FACTOR = 0.2

RECYCLER_SPEED = 0.5
RECYCLER_SLOTS = 4


@dataclass(frozen=True)
class Machine:
    """A crafting machine's base stats."""

    name: str
    speed: float
    slots: int
    prod_bonus: float
    icon: str


MACHINES = {
    "assembling-machine-3": Machine("Assembling machine 3", 1.25, 4, 0.0, "assembling_machine_3"),
    "electric-furnace": Machine("Electric furnace", 2.0, 2, 0.0, "electric_furnace"),
    "chemical-plant": Machine("Chemical plant", 1.0, 3, 0.0, "chemical_plant"),
    "centrifuge": Machine("Centrifuge", 1.0, 2, 0.0, "centrifuge"),
    "electromagnetic-plant": Machine("Electromagnetic plant", 2.0, 5, 0.5, "electromagnetic_plant"),
    "foundry": Machine("Foundry", 4.0, 4, 0.5, "foundry"),
    "biochamber": Machine("Biochamber", 2.0, 4, 0.5, "biochamber"),
    "cryogenic-plant": Machine("Cryogenic plant", 2.0, 8, 0.0, "cryogenic_plant"),
}

# Bonus per module, indexed by module quality (Normal → Legendary)
QUALITY_MODULE_BONUS = {
    1: [0.010, 0.013, 0.016, 0.019, 0.025],
    2: [0.020, 0.026, 0.032, 0.038, 0.050],
    3: [0.025, 0.032, 0.040, 0.047, 0.0625],
}
PROD_MODULE_BONUS = {
    1: [0.04, 0.05, 0.06, 0.07, 0.10],
    2: [0.06, 0.07, 0.09, 0.11, 0.15],
    3: [0.10, 0.13, 0.16, 0.19, 0.25],
}


def machine_speed(base_speed: float, machine_quality: int, speed_penalty: float) -> float:
    """Effective crafting speed of a machine after its quality bonus and module penalties."""
    assert 0 <= machine_quality < NUM_TIERS
    return base_speed * QUALITY_SPEED_MULTIPLIER[machine_quality] * max(1 - speed_penalty, MIN_SPEED_FACTOR)


def recycler_speed(recycler_quality: int, num_quality_modules: int = RECYCLER_SLOTS) -> float:
    """Effective speed of a recycler filled with quality modules."""
    return machine_speed(RECYCLER_SPEED, recycler_quality, num_quality_modules * QUALITY_MODULE_SPEED_PENALTY)


def game_data() -> dict:
    """Serialisable summary of the game data, for the frontend's dropdowns."""
    return {
        "machines": {
            key: {"name": m.name, "speed": m.speed, "slots": m.slots, "prod_bonus": m.prod_bonus, "icon": m.icon}
            for key, m in MACHINES.items()
        },
        "quality_module_bonus": QUALITY_MODULE_BONUS,
        "prod_module_bonus": PROD_MODULE_BONUS,
        "recycler_slots": RECYCLER_SLOTS,
        **recipe_data(),
    }
