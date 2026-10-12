"""Factorio: Space Age recipe data, taken from the wiki (https://wiki.factorio.com)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """An item's display name and icon."""

    name: str
    icon: str


@dataclass(frozen=True)
class Recipe:
    """A recipe's inputs, outputs and the machines (keys of MACHINES) that can craft it."""

    name: str
    time: float
    ingredients: dict[str, float]
    products: dict[str, float]
    machines: tuple[str, ...]

    def ratio(self, ingredient: str) -> float:
        """Products per unit of one ingredient, as used by the assembler loop."""
        return sum(self.products.values()) / self.ingredients[ingredient]


ITEMS = {
    "advanced-circuit": Item("Advanced circuit", "advanced_circuit"),
    "carbon": Item("Carbon", "carbon"),
    "carbon-fiber": Item("Carbon fiber", "carbon_fiber"),
    "coal": Item("Coal", "coal"),
    "copper-plate": Item("Copper plate", "copper_plate"),
    "electromagnetic-plant": Item("Electromagnetic plant", "electromagnetic_plant"),
    "fast-underground-belt": Item("Fast underground belt", "fast_underground_belt"),
    "holmium-plate": Item("Holmium plate", "holmium_plate"),
    "iron-gear-wheel": Item("Iron gear wheel", "iron_gear_wheel"),
    "iron-plate": Item("Iron plate", "iron_plate"),
    "low-density-structure": Item("Low density structure", "low_density_structure"),
    "molten-copper": Item("Molten copper", "molten_copper"),
    "molten-iron": Item("Molten iron", "molten_iron"),
    "pipe": Item("Pipe", "pipe"),
    "pipe-to-ground": Item("Pipe to ground", "pipe_to_ground"),
    "plastic-bar": Item("Plastic bar", "plastic_bar"),
    "processing-unit": Item("Processing unit", "processing_unit"),
    "refined-concrete": Item("Refined concrete", "refined_concrete"),
    "steel-plate": Item("Steel plate", "steel_plate"),
    "sulfur": Item("Sulfur", "sulfur"),
    "toolbelt-equipment": Item("Toolbelt equipment", "toolbelt_equipment"),
    "underground-belt": Item("Underground belt", "underground_belt"),
    "water": Item("Water", "water"),
    "yumako-mash": Item("Yumako mash", "yumako_mash"),
}

ASSEMBLERS = ("assembling-machine-3",)

RECIPES = {
    "casting-low-density-structure": Recipe(
        "Casting low density structure",
        time=15,
        ingredients={"molten-iron": 80, "molten-copper": 250, "plastic-bar": 5},
        products={"low-density-structure": 1},
        machines=("foundry",),
    ),
    "casting-pipe": Recipe(
        "Casting pipe",
        time=1,
        ingredients={"molten-iron": 10},
        products={"pipe": 1},
        machines=("foundry",),
    ),
    "casting-pipe-to-ground": Recipe(
        "Casting pipe to ground",
        time=1,
        ingredients={"pipe": 10, "molten-iron": 50},
        products={"pipe-to-ground": 2},
        machines=("foundry",),
    ),
    "coal-synthesis": Recipe(
        "Coal synthesis",
        time=2,
        ingredients={"sulfur": 1, "carbon": 5, "water": 10},
        products={"coal": 1},
        machines=("chemical-plant",),
    ),
    "electromagnetic-plant": Recipe(
        "Electromagnetic plant",
        time=10,
        ingredients={"holmium-plate": 150, "processing-unit": 50, "refined-concrete": 50, "steel-plate": 50},
        products={"electromagnetic-plant": 1},
        machines=(*ASSEMBLERS, "electromagnetic-plant"),
    ),
    "fast-underground-belt": Recipe(
        "Fast underground belt",
        time=2,
        ingredients={"iron-gear-wheel": 40, "underground-belt": 2},
        products={"fast-underground-belt": 2},
        machines=(*ASSEMBLERS, "foundry"),
    ),
    "low-density-structure": Recipe(
        "Low density structure",
        time=15,
        ingredients={"copper-plate": 20, "steel-plate": 2, "plastic-bar": 5},
        products={"low-density-structure": 1},
        machines=ASSEMBLERS,
    ),
    "pipe": Recipe(
        "Pipe",
        time=0.5,
        ingredients={"iron-plate": 1},
        products={"pipe": 1},
        machines=ASSEMBLERS,
    ),
    "pipe-to-ground": Recipe(
        "Pipe to ground",
        time=0.5,
        ingredients={"iron-plate": 5, "pipe": 10},
        products={"pipe-to-ground": 2},
        machines=ASSEMBLERS,
    ),
    "carbon-fiber": Recipe(
        "Carbon fiber",
        time=5,
        ingredients={"carbon": 1, "yumako-mash": 10},
        products={"carbon-fiber": 1},
        machines=("biochamber",),
    ),
    "toolbelt-equipment": Recipe(
        "Toolbelt equipment",
        time=10,
        ingredients={"advanced-circuit": 3, "carbon-fiber": 10},
        products={"toolbelt-equipment": 1},
        machines=ASSEMBLERS,
    ),
}


def recipe_data() -> dict:
    """Serialisable recipes and items, for the frontend's recipe picker."""
    return {
        "items": {key: {"name": i.name, "icon": i.icon} for key, i in ITEMS.items()},
        "recipes": {
            key: {
                "name": r.name,
                "time": r.time,
                "ingredients": r.ingredients,
                "products": r.products,
                "machines": list(r.machines),
                "icon": ITEMS[next(iter(r.products))].icon,
            }
            for key, r in RECIPES.items()
        },
    }
