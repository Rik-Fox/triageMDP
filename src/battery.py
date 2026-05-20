from typing import Dict, List
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from util import HealthState, Product, Action, draw_labeled_multigraph
from model import CEOption


class Cell(Product):
    """Leaf node component."""

    def __init__(self, health=1.0, name: str = "Cell") -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
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

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates a dictionary of ground-truth health values for the product and its components."""
        return {self.name: self.health_state.product_health}


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

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates a dictionary of ground-truth health values for the product and its components."""
        state = {self.name: self.health_state.product_health}
        for component in self.components:
            state.update(component.generate_true_state(mode))
        return state


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

        self.ce_options = [
            CEOption(
                "Reuse_EV_Pack",
                base_value=800.0,
                # High risk: Ramp up from 0.80, Safe Zone 0.90 to 1.0
                fuzzy_params=(0.70, 0.80, 1.0, 1.0),
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
                fuzzy_params=(0.0, 0.4, 1.0, 1.0),
                prerequisites=["High_Voltage_Isolation"],
            ),
        ]

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates ground-truth health by enforcing hierarchical averages."""
        # 1. Define Cell Healths (Base level)
        if mode == "uniform":
            # All modules are borderline (0.80)
            cell_healths = [[0.80, 0.80], [0.80, 0.80], [0.80, 0.80]]
        elif mode == "moderate":
            # Slight variance (0.90, 0.80, 0.70)
            cell_healths = [[0.90, 0.90], [0.80, 0.80], [0.70, 0.70]]
        elif mode == "extreme":
            # Massive variance (1.0, 1.0, 0.40) -> Two perfect modules, one dead module
            cell_healths = [[1.00, 1.00], [1.00, 1.00], [0.40, 0.40]]
        else:  # Random
            cell_healths = [np.random.uniform(0.4, 1.0, 2).tolist() for _ in range(3)]

        state = {}
        module_healths = []

        # 2. Build Modules from average of Cells
        for i, cells in enumerate(cell_healths):
            m_health = float(np.mean(cells))
            module_healths.append(m_health)
            state[f"Module_{i}"] = m_health
            for j, c_health in enumerate(cells):
                state[f"Module_{i}_Cell_{j}"] = c_health

        # 3. Build Pack from average of Modules
        pack_health = float(np.mean(module_healths))
        state[self.name] = pack_health

        return state


if __name__ == "__main__":
    from model import StateAugmentedDisassemblyGraph, Solver

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
    sag = StateAugmentedDisassemblyGraph(battery)
    solver = Solver()
    policy = solver.solve(sag.state_map[sag.root_key], obs)

    # 4. Hierarchical Output
    print("OPTIMIZED TRIAGE DECISION TREE:")

    id_to_node_map = {node.id: node for node in sag.state_map.values()}

    def print_decision(node_id, indent=""):
        if node_id not in policy:
            return
        action = policy[node_id]
        print(f"{indent}[{node_id}] -> {action}")

        # If the decision was to disassemble, print the children's decisions
        if "Disassemble" in action or "Extract" in action:
            node = id_to_node_map.get(node_id)
            if node:
                action_name = action.split("Action: ")[1]
                for act_obj, child_nodes_list in node.children:
                    if act_obj.name == action_name:
                        for child_node in child_nodes_list:
                            print_decision(child_node.id, indent + "    |-- ")
                        break

    root_node_id = sag.state_map[sag.root_key].id
    print_decision(root_node_id)
    print("=" * 70 + "\n")
