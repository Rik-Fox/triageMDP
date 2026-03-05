from typing import Dict, List
import matplotlib.pyplot as plt
import networkx as nx
from util import HealthState, Product, Action, draw_labeled_multigraph
from model import CEOption


class Cell(Product):
    """Leaf node component."""

    def __init__(self, health=1.0, name: str = "Cell") -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        # old CE options with sigmoid fuzziness
        # # Define Circular Economy Options for a Cell
        # self.ce_options = [
        #     CEOption(
        #         name="Reuse_Cell",
        #         base_value=50.0,  # High value
        #         fuzziness_k=15.0,  # Strict quality curve
        #         threshold=0.85,  # High threshold
        #     ),
        #     CEOption(
        #         name="Recycle_Materials",
        #         base_value=5.0,  # Low value
        #         fuzziness_k=1.0,  # Flat curve (easy to qualify)
        #         threshold=0.1,
        #     ),
        # ]

        # Define Circular Economy Options for a Cell
        self.ce_options = [
            CEOption(
                name="Reuse_Cell",
                base_value=50.0,
                # Ramp up from 0.75, Safe Zone 0.85 to 1.0
                fuzzy_params=(0.75, 0.85, 1.0, 1.0),
            ),
            CEOption(
                name="Recycle_Materials",
                base_value=5.0,
                # Flat acceptance curve
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
            ),
        ]


class Module(Product):
    """Intermediate assembly."""

    def __init__(
        self, health=1.0, cell_healths: List[float] = None, name: str = "Module"
    ) -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        if cell_healths is None:
            cell_healths = [1.0, 1.0]

        self.components = [
            Cell(health=h, name=f"{name}_Cell_{i}") for i, h in enumerate(cell_healths)
        ]

        # Requires Bench Diagnostic before extraction
        self.actions = [
            Action("Bench_Diagnostic", cost=10.0, time=5.0),
            Action(
                "Extract_Cells",
                cost=15.0,
                time=10.0,
                prerequisites=["Bench_Diagnostic"],
                is_disassembly=True,
            ),
        ]

        # old CE options with sigmoid fuzziness
        # self.ce_options = [
        #     CEOption(
        #         "Reuse_Module",
        #         base_value=120.0,
        #         fuzziness_k=12.0,
        #         threshold=0.8,
        #         prerequisites=["Bench_Diagnostic"],
        #     ),
        #     CEOption(
        #         "Recycle_Module_Direct", base_value=10.0, fuzziness_k=1.0, threshold=0.0
        #     ),
        # ]

        self.ce_options = [
            CEOption(
                "Reuse_Module",
                base_value=120.0,
                # Ramp up from 0.70, Safe Zone 0.80 to 1.0
                fuzzy_params=(0.70, 0.80, 1.0, 1.0),
                prerequisites=["Bench_Diagnostic"],
            ),
            CEOption(
                "Recycle_Module_Direct",
                base_value=10.0,
                # Flat acceptance curve
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
            ),
        ]


class Battery(Product):
    """Root assembly."""

    def __init__(
        self, health=1.0, module_healths: List[float] = None, name="Battery_Pack"
    ) -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        if module_healths is None:
            module_healths = [1.0, 1.0, 1.0]

        self.components = [
            Module(health=h, name=f"Module_{i}") for i, h in enumerate(module_healths)
        ]

        # Meaningful Sequence Constraints
        self.actions = [
            Action("High_Voltage_Isolation", cost=10.0, time=5.0),
            Action(
                "Remove_Top_Cover",
                cost=3.0,
                time=2.0,
                prerequisites=["High_Voltage_Isolation"],
            ),
            Action(
                "Remove_Thermal_Shield",
                cost=3.0,
                time=2.0,
                prerequisites=["Remove_Top_Cover"],
            ),
            Action(
                "Extract_Modules",
                cost=30.0,
                time=20.0,
                prerequisites=["Remove_Top_Cover"],
                is_disassembly=True,
            ),
        ]

        # old CE options with sigmoid fuzziness
        # self.ce_options = [
        #     CEOption(
        #         "Reuse_EV_Pack",
        #         base_value=800.0,
        #         fuzziness_k=20.0,
        #         threshold=0.9,
        #         prerequisites=["High_Voltage_Isolation"],
        #     ),
        #     CEOption(
        #         "Repurpose_Storage",
        #         base_value=400.0,
        #         fuzziness_k=10.0,
        #         threshold=0.7,
        #         prerequisites=["Remove_Thermal_Shield"],
        #     ),
        #     CEOption(
        #         "Recycle_Pack_Direct",
        #         base_value=50.0,
        #         fuzziness_k=1.0,
        #         threshold=0.0,
        #         prerequisites=["High_Voltage_Isolation"],
        #     ),
        # ]

        self.ce_options = [
            CEOption(
                "Reuse_EV_Pack",
                base_value=800.0,
                # High risk: Ramp up from 0.80, Safe Zone 0.90 to 1.0
                fuzzy_params=(0.80, 0.90, 1.0, 1.0),
                prerequisites=["High_Voltage_Isolation"],
            ),
            CEOption(
                "Repurpose_Storage",
                base_value=400.0,
                # Lower risk: Ramp up from 0.55, Safe Zone 0.70 to 1.0
                fuzzy_params=(0.55, 0.70, 1.0, 1.0),
                prerequisites=["Remove_Thermal_Shield"],
            ),
            CEOption(
                "Recycle_Pack_Direct",
                base_value=50.0,
                # Flat acceptance curve
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
                prerequisites=["High_Voltage_Isolation"],
            ),
        ]


if __name__ == "__main__":
    from model import StateAugmentedGraph, Solver

    print("=" * 70)
    print("TEST 3: Integrated Battery Triage Simulation")
    print("-" * 70)
    print("SCENARIO:")
    print("  We are triaging a full EV Battery Pack containing 2 Modules.")
    print("  Extracting modules costs £30. Extracting cells costs £15.")
    print("\nINPUT OBSERVATIONS:")
    print("  - Battery Pack Overall: 0.50 (Poor - cannot be reused directly)")
    print("  - Module 0: 0.40 (Poor)")
    print("  - Module 1: 0.95 (Excellent)")
    print("\nEXPECTED OUTCOME:")
    print("  The solver should realize the pack is too degraded for full reuse.")
    print("  It should pay the £30 to disassemble it.")
    print("  It should Recycle Module 0 (too degraded).")
    print(
        "  It should attempt to Reuse Module 1 (excellent health justifies extraction)."
    )
    print("-" * 70)

    # 1. Create the degraded battery
    battery = Battery(health=0.5, module_healths=[0.4, 0.95], name="Pack_X1")

    # 2. Define the Noisy Observations
    obs = {
        "Pack_X1": 0.5,
        "Module_0": 0.4,
        "Module_1": 0.95,
    }

    # 3. Build & Solve
    sag = StateAugmentedGraph(battery)
    solver = Solver()
    policy = solver.solve(sag.root, obs)

    # 4. Hierarchical Output
    print("OPTIMIZED TRIAGE DECISION TREE:")

    def print_decision(node_id, indent=""):
        if node_id not in policy:
            return
        action = policy[node_id]
        print(f"{indent}[{node_id}] -> {action}")

        # If the decision was to disassemble, print the children's decisions
        if "Disassemble" in action or "Extract" in action:
            node = sag.product_map.get(node_id)
            if node:
                for _, child in node.children:
                    print_decision(child.id, indent + "    |-- ")

    print_decision(battery.name)
    print("=" * 70 + "\n")
