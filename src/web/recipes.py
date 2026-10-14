"""Factorio: Space Age recipe data, taken from the wiki (https://wiki.factorio.com)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """An item's display name and icon."""

    name: str
    icon: str


@dataclass(frozen=True)
class Recipe:
    """A recipe's inputs, outputs, the machines (keys of MACHINES) that can craft it, and its in-game tab."""

    name: str
    time: float
    ingredients: dict[str, float]
    products: dict[str, float]
    machines: tuple[str, ...]
    group: str

    def ratio(self, ingredient: str) -> float:
        """Products per unit of one ingredient, as used by the assembler loop."""
        return sum(self.products.values()) / self.ingredients[ingredient]


ITEMS = {
    "advanced-circuit": Item("Advanced circuit", "advanced_circuit"),
    "ammonia": Item("Ammonia", "ammonia"),
    "battery": Item("Battery", "battery"),
    "carbon": Item("Carbon", "carbon"),
    "carbon-fiber": Item("Carbon fiber", "carbon_fiber"),
    "coal": Item("Coal", "coal"),
    "copper-cable": Item("Copper cable", "copper_cable"),
    "copper-plate": Item("Copper plate", "copper_plate"),
    "electrolyte": Item("Electrolyte", "electrolyte"),
    "electromagnetic-plant": Item("Electromagnetic plant", "electromagnetic_plant"),
    "electronic-circuit": Item("Electronic circuit", "electronic_circuit"),
    "fast-underground-belt": Item("Fast underground belt", "fast_underground_belt"),
    "fluorine": Item("Fluorine", "fluorine"),
    "fluoroketone-cold": Item("Fluoroketone (cold)", "fluoroketone_cold"),
    "fluoroketone-hot": Item("Fluoroketone (hot)", "fluoroketone_hot"),
    "holmium-plate": Item("Holmium plate", "holmium_plate"),
    "iron-gear-wheel": Item("Iron gear wheel", "iron_gear_wheel"),
    "iron-plate": Item("Iron plate", "iron_plate"),
    "light-oil": Item("Light oil", "light_oil"),
    "lithium": Item("Lithium", "lithium"),
    "lithium-brine": Item("Lithium brine", "lithium_brine"),
    "lithium-plate": Item("Lithium plate", "lithium_plate"),
    "low-density-structure": Item("Low density structure", "low_density_structure"),
    "molten-copper": Item("Molten copper", "molten_copper"),
    "molten-iron": Item("Molten iron", "molten_iron"),
    "pipe": Item("Pipe", "pipe"),
    "pipe-to-ground": Item("Pipe to ground", "pipe_to_ground"),
    "plastic-bar": Item("Plastic bar", "plastic_bar"),
    "processing-unit": Item("Processing unit", "processing_unit"),
    "quantum-processor": Item("Quantum processor", "quantum_processor"),
    "refined-concrete": Item("Refined concrete", "refined_concrete"),
    "solid-fuel": Item("Solid fuel", "solid_fuel"),
    "steel-plate": Item("Steel plate", "steel_plate"),
    "sulfur": Item("Sulfur", "sulfur"),
    "sulfuric-acid": Item("Sulfuric acid", "sulfuric_acid"),
    "supercapacitor": Item("Supercapacitor", "supercapacitor"),
    "superconductor": Item("Superconductor", "superconductor"),
    "toolbelt-equipment": Item("Toolbelt equipment", "toolbelt_equipment"),
    "tungsten-carbide": Item("Tungsten carbide", "tungsten_carbide"),
    "tungsten-ore": Item("Tungsten ore", "tungsten_ore"),
    "underground-belt": Item("Underground belt", "underground_belt"),
    "water": Item("Water", "water"),
    "yumako-mash": Item("Yumako mash", "yumako_mash"),
}

ASSEMBLERS = ("assembling-machine-3",)

# The in-game crafting tabs, in the order the recipe picker lists them
GROUPS = ("Production", "Logistics", "Intermediate", "Combat", "Fluids")

RECIPES = {
    "advanced-circuit": Recipe(
        "Advanced circuit",
        time=6,
        ingredients={"copper-cable": 4, "electronic-circuit": 2, "plastic-bar": 2},
        products={"advanced-circuit": 1},
        machines=(*ASSEMBLERS, "electromagnetic-plant"),
        group="Intermediate",
    ),
    "casting-copper-cable": Recipe(
        "Casting copper cable",
        time=1,
        ingredients={"molten-copper": 5},
        products={"copper-cable": 2},
        machines=("foundry",),
        group="Intermediate",
    ),
    "copper-cable": Recipe(
        "Copper cable",
        time=0.5,
        ingredients={"copper-plate": 1},
        products={"copper-cable": 2},
        machines=(*ASSEMBLERS, "electromagnetic-plant"),
        group="Intermediate",
    ),
    "electronic-circuit": Recipe(
        "Electronic circuit",
        time=0.5,
        ingredients={"copper-cable": 3, "iron-plate": 1},
        products={"electronic-circuit": 1},
        machines=(*ASSEMBLERS, "electromagnetic-plant"),
        group="Intermediate",
    ),
    "processing-unit": Recipe(
        "Processing unit",
        time=10,
        ingredients={"electronic-circuit": 20, "advanced-circuit": 2, "sulfuric-acid": 5},
        products={"processing-unit": 1},
        machines=(*ASSEMBLERS, "electromagnetic-plant"),
        group="Intermediate",
    ),
    "quantum-processor": Recipe(
        "Quantum processor",
        time=30,
        ingredients={
            "carbon-fiber": 1,
            "fluoroketone-cold": 10,
            "lithium-plate": 2,
            "processing-unit": 1,
            "superconductor": 1,
            "tungsten-carbide": 1,
        },
        products={"quantum-processor": 1, "fluoroketone-hot": 5},
        machines=("electromagnetic-plant",),
        group="Intermediate",
    ),
    "casting-low-density-structure": Recipe(
        "Casting low density structure",
        time=15,
        ingredients={"molten-iron": 80, "molten-copper": 250, "plastic-bar": 5},
        products={"low-density-structure": 1},
        machines=("foundry",),
        group="Intermediate",
    ),
    "casting-pipe": Recipe(
        "Casting pipe",
        time=1,
        ingredients={"molten-iron": 10},
        products={"pipe": 1},
        machines=("foundry",),
        group="Logistics",
    ),
    "casting-pipe-to-ground": Recipe(
        "Casting pipe to ground",
        time=1,
        ingredients={"pipe": 10, "molten-iron": 50},
        products={"pipe-to-ground": 2},
        machines=("foundry",),
        group="Logistics",
    ),
    "coal-synthesis": Recipe(
        "Coal synthesis",
        time=2,
        ingredients={"sulfur": 1, "carbon": 5, "water": 10},
        products={"coal": 1},
        machines=("chemical-plant",),
        group="Intermediate",
    ),
    "electromagnetic-plant": Recipe(
        "Electromagnetic plant",
        time=10,
        ingredients={"holmium-plate": 150, "processing-unit": 50, "refined-concrete": 50, "steel-plate": 50},
        products={"electromagnetic-plant": 1},
        machines=(*ASSEMBLERS, "electromagnetic-plant"),
        group="Production",
    ),
    "fast-underground-belt": Recipe(
        "Fast underground belt",
        time=2,
        ingredients={"iron-gear-wheel": 40, "underground-belt": 2},
        products={"fast-underground-belt": 2},
        machines=(*ASSEMBLERS, "foundry"),
        group="Logistics",
    ),
    "low-density-structure": Recipe(
        "Low density structure",
        time=15,
        ingredients={"copper-plate": 20, "steel-plate": 2, "plastic-bar": 5},
        products={"low-density-structure": 1},
        machines=ASSEMBLERS,
        group="Intermediate",
    ),
    "pipe": Recipe(
        "Pipe",
        time=0.5,
        ingredients={"iron-plate": 1},
        products={"pipe": 1},
        machines=ASSEMBLERS,
        group="Logistics",
    ),
    "pipe-to-ground": Recipe(
        "Pipe to ground",
        time=0.5,
        ingredients={"iron-plate": 5, "pipe": 10},
        products={"pipe-to-ground": 2},
        machines=ASSEMBLERS,
        group="Logistics",
    ),
    "battery": Recipe(
        "Battery",
        time=4,
        ingredients={"sulfuric-acid": 20, "iron-plate": 1, "copper-plate": 1},
        products={"battery": 1},
        machines=("chemical-plant", "cryogenic-plant"),
        group="Intermediate",
    ),
    "fluoroketone-cold": Recipe(
        "Fluoroketone (cold)",
        time=5,
        ingredients={"fluoroketone-hot": 10},
        products={"fluoroketone-cold": 10},
        machines=("cryogenic-plant",),
        group="Fluids",
    ),
    "fluoroketone-hot": Recipe(
        "Fluoroketone (hot)",
        time=10,
        ingredients={"ammonia": 50, "fluorine": 50, "lithium": 1, "solid-fuel": 1},
        products={"fluoroketone-hot": 50},
        machines=("cryogenic-plant",),
        group="Fluids",
    ),
    "lithium": Recipe(
        "Lithium",
        time=20,
        ingredients={"ammonia": 50, "holmium-plate": 1, "lithium-brine": 50},
        products={"lithium": 5},
        machines=("chemical-plant", "cryogenic-plant"),
        group="Intermediate",
    ),
    "lithium-plate": Recipe(
        "Lithium plate",
        time=6.4,
        ingredients={"lithium": 1},
        products={"lithium-plate": 1},
        machines=("electric-furnace",),
        group="Intermediate",
    ),
    "sulfuric-acid": Recipe(
        "Sulfuric acid",
        time=1,
        ingredients={"iron-plate": 1, "sulfur": 5, "water": 100},
        products={"sulfuric-acid": 50},
        machines=("chemical-plant",),
        group="Fluids",
    ),
    "supercapacitor": Recipe(
        "Supercapacitor",
        time=10,
        ingredients={"battery": 1, "electrolyte": 10, "electronic-circuit": 4, "holmium-plate": 2, "superconductor": 2},
        products={"supercapacitor": 1},
        machines=("electromagnetic-plant",),
        group="Intermediate",
    ),
    "superconductor": Recipe(
        "Superconductor",
        time=5,
        ingredients={"copper-plate": 1, "holmium-plate": 1, "light-oil": 5, "plastic-bar": 1},
        products={"superconductor": 2},
        machines=("electromagnetic-plant",),
        group="Intermediate",
    ),
    "tungsten-carbide": Recipe(
        "Tungsten carbide",
        time=1,
        ingredients={"carbon": 1, "sulfuric-acid": 10, "tungsten-ore": 2},
        products={"tungsten-carbide": 1},
        machines=ASSEMBLERS,
        group="Intermediate",
    ),
    "carbon-fiber": Recipe(
        "Carbon fiber",
        time=5,
        ingredients={"carbon": 1, "yumako-mash": 10},
        products={"carbon-fiber": 1},
        machines=("biochamber",),
        group="Intermediate",
    ),
    "toolbelt-equipment": Recipe(
        "Toolbelt equipment",
        time=10,
        ingredients={"advanced-circuit": 3, "carbon-fiber": 10},
        products={"toolbelt-equipment": 1},
        machines=ASSEMBLERS,
        group="Combat",
    ),
}


def recipe_data() -> dict:
    """Serialisable recipes and items, for the frontend's recipe picker."""
    return {
        "items": {key: {"name": i.name, "icon": i.icon} for key, i in ITEMS.items()},
        "groups": list(GROUPS),
        "recipes": {
            key: {
                "name": r.name,
                "time": r.time,
                "ingredients": r.ingredients,
                "products": r.products,
                "machines": list(r.machines),
                "group": r.group,
                "icon": ITEMS[next(iter(r.products))].icon,
            }
            for key, r in RECIPES.items()
        },
    }
