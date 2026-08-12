from typing import Dict, List
import numpy as np

# from src.util import HealthState, Product, Action, draw_labeled_multigraph
# from src.model import CEOption

from util import HealthState, Product, Action, draw_labeled_multigraph
from model import CEOption


class CPU(Product):
    """Intermediate assembly."""

    def __init__(self, health=1.0, value=50, f=0.5, name: str = "CPU") -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)

        self.actions = [
            Action(
                "Bench_Diagnostic_CPU",
                cost=3.0,  # Reduced cost
                time=1.0,
                prerequisites=["Power_On_Test", "Bench_Diagnostic_Motherboard"],
            ),
        ]
        self.ce_options = [
            CEOption(
                "Reuse_CPU",
                base_value=value * health * 0.95,
                phi_k=0.7 + (np.random.randn() * 0.2),
                E_k=0.0,
                Phi=1.0,
                fuzzy_params=(0.7, 0.8, 1.0, 1.0),
                prerequisites=["Bench_Diagnostic_CPU"],
            ),
            CEOption(
                "Repair_CPU",
                base_value=value * health * 0.5,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)) * f,
                Phi=0.8,
                fuzzy_params=(0.4, 0.6, 0.8, 0.9),
                prerequisites=["Bench_Diagnostic_CPU"],
            ),
            CEOption(
                "Recycle_CPU",
                base_value=value * health * 0.3,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(2.0 + np.random.randn() * (2.0 * 0.2)),
                Phi=0.5,
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
            ),
        ]
        self.value = value
        self.value_weight = f

        self._build_graph()

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates a dictionary of ground-truth health values for the product and its components."""
        return {self.name: self.health_state.product_health}


class GPU(Product):
    """Intermediate assembly."""

    def __init__(self, health=1.0, value=100, f=0.5, name: str = "GPU") -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        self.actions = [
            Action(
                "Bench_Diagnostic_GPU",
                cost=5.0,  # Reduced cost
                time=2.0,
                prerequisites=["Power_On_Test", "Bench_Diagnostic_Motherboard"],
            ),
        ]
        self.ce_options = [
            CEOption(
                "Reuse_GPU",
                base_value=value * health * 0.95,
                phi_k=0.7 + (np.random.randn() * 0.2),
                E_k=0.0,
                Phi=1.0,
                fuzzy_params=(0.65, 0.75, 1.0, 1.0),
                prerequisites=["Bench_Diagnostic_GPU"],
            ),
            CEOption(
                "Repair_GPU",
                base_value=value * health * 0.5,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)) * f,
                Phi=0.8,
                fuzzy_params=(0.4, 0.5, 0.7, 0.8),
                prerequisites=["Bench_Diagnostic_GPU"],
            ),
            CEOption(
                "Recycle_GPU",
                base_value=value * health * 0.3,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(2.0 + np.random.randn() * (2.0 * 0.2)),
                Phi=0.5,
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
            ),
        ]
        self.value = value
        self.value_weight = f

        self._build_graph()

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates a dictionary of ground-truth health values for the product and its components."""
        return {self.name: self.health_state.product_health}


class Memory(Product):
    """Leaf node component."""

    def __init__(self, health=1.0, value=50, f=0.5, name: str = "Memory") -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        self.actions = [
            Action(
                "Bench_Diagnostic_Memory",
                cost=4.0,  # Reduced cost
                time=1.0,
                prerequisites=["Power_On_Test", "Bench_Diagnostic_Motherboard"],
            ),
        ]
        self.ce_options = [
            CEOption(
                name="Reuse_Memory",
                base_value=value * health * 0.95,
                phi_k=0.7 + (np.random.randn() * 0.2),
                E_k=0.0,
                Phi=1.0,
                fuzzy_params=(0.8, 0.9, 1.0, 1.0),
                prerequisites=["Bench_Diagnostic_Memory"],
            ),
            CEOption(
                name="Repair_Memory",
                base_value=value * health * 0.5,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)) * f,
                Phi=0.8,
                fuzzy_params=(0.5, 0.6, 0.8, 0.9),
                prerequisites=["Bench_Diagnostic_Memory"],
            ),
            CEOption(
                name="Recycle_Memory",
                base_value=value * health * 0.3,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(2.0 + np.random.randn() * (2.0 * 0.2)),
                Phi=0.5,
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
            ),
        ]
        self.value = value
        self.value_weight = f

        self._build_graph()

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates a dictionary of ground-truth health values for the product and its components."""
        return {self.name: self.health_state.product_health}


class DataStorage(Product):
    """Leaf node component."""

    def __init__(self, health=1.0, value=250, f=0.5, name: str = "DataStorage") -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        self.actions = self.actions = [
            Action(
                "Bench_Diagnostic_DataStorage",
                cost=8.0,  # Reduced cost
                time=5.0,
                prerequisites=["Power_On_Test", "Bench_Diagnostic_Motherboard"],
            ),
            Action(
                "Wipe_DataStorage", cost=5.0, time=15.0, prerequisites=["Open_Chassis"]
            ),
        ]
        self.ce_options = [
            CEOption(
                name="Reuse_DataStorage",
                base_value=value * health * 0.95,
                phi_k=0.7 + (np.random.randn() * 0.2),
                E_k=0.0,
                Phi=1.0,
                fuzzy_params=(0.7, 0.8, 1.0, 1.0),
                prerequisites=["Bench_Diagnostic_DataStorage", "Wipe_DataStorage"],
            ),
            CEOption(
                name="Repair_DataStorage",
                base_value=value * health * 0.5,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)) * f,
                Phi=0.8,
                fuzzy_params=(0.4, 0.5, 0.8, 0.9),
                prerequisites=["In_Situ_Diagnostic_DataStorage", "Wipe_DataStorage"],
            ),
            CEOption(
                name="Recycle_DataStorage",
                base_value=value * health * 0.3,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(2.0 + np.random.randn() * (2.0 * 0.2)),
                Phi=0.5,
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
                prerequisites=["Wipe_DataStorage"],
            ),
        ]
        self.value = value
        self.value_weight = f
        self._build_graph()

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates a dictionary of ground-truth health values for the product and its components."""
        return {self.name: self.health_state.product_health}


class Motherboard(Product):
    """Intermediate assembly."""

    def __init__(
        self,
        health=1.0,
        value=500,
        f=1.0,
        cpu_health: float = 1.0,
        gpu_health: float = 1.0,
        memory_health: float = 1.0,
        storage_health: float = 1.0,
        name: str = "Motherboard",
    ) -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        self.components = [
            CPU(
                health=cpu_health,
                value=(value / f) * 0.05,
                f=0.05,
                name=f"{name}_CPU",
            ),  # value divided by motherboard f returns full value as emisions weigths are percent of whole product
            GPU(
                health=gpu_health,
                value=(value / f) * 0.1,
                f=0.1,
                name=f"{name}_GPU",
            ),
            Memory(
                health=memory_health,
                value=(value / f) * 0.05,
                f=0.05,
                name=f"{name}_Memory",
            ),
            DataStorage(
                health=storage_health,
                value=(value / f) * 0.25,
                f=0.25,
                name=f"{name}_DataStorage",
            ),
        ]

        self.actions = [
            Action("Full_System_Diagnostic", cost=20.0, time=15.0),
            Action("Bench_Diagnostic_Motherboard", cost=15.0, time=10.0),
            Action(
                "In_Situ_Diagnostic_CPU",
                cost=2.0,
                time=2.0,
                prerequisites=["Power_On_Test"],
            ),
            Action(
                "In_Situ_Diagnostic_GPU",
                cost=2.0,
                time=2.0,
                prerequisites=["Power_On_Test"],
            ),
            Action(
                "In_Situ_Diagnostic_Memory",
                cost=2.0,
                time=2.0,
                prerequisites=["Power_On_Test"],
            ),
            Action(
                "In_Situ_Diagnostic_DataStorage",
                cost=2.0,
                time=2.0,
                prerequisites=["Power_On_Test"],
            ),
            Action(
                "Extract_Components",
                cost=10.0,  # Significantly reduced cost to incentivize disassembly
                time=10.0,
                prerequisites=["Bench_Diagnostic_Motherboard"],
                is_disassembly=True,
            ),
        ]
        self.ce_options = [
            CEOption(
                "Reuse_Motherboard",
                base_value=value * health * 1.0,
                phi_k=0.7 + (np.random.randn() * 0.2),
                E_k=0.0,
                Phi=1.0,
                fuzzy_params=(0.75, 0.85, 1.0, 1.0),
                prerequisites=[
                    "Bench_Diagnostic_Motherboard",
                    "Full_System_Diagnostic",
                ],
            ),
            CEOption(
                "Repair_Motherboard",
                base_value=value * health * 0.6,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)) * f,
                Phi=0.8,
                fuzzy_params=(0.4, 0.5, 0.6, 0.7),
                prerequisites=[
                    "Bench_Diagnostic_Motherboard",
                    "Full_System_Diagnostic",
                ],  # TODO: incorporate costs of extracting components and replacing them
                # (refurb is for other fixes than the resellable components)
            ),
            CEOption(
                "Refurbish_Motherboard",
                base_value=value * health * 0.7,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)) * f,
                Phi=0.8,
                fuzzy_params=(0.6, 0.7, 0.8, 0.9),
                prerequisites=[
                    "Bench_Diagnostic_Motherboard",
                    "In_Situ_Diagnostic_CPU",
                    "In_Situ_Diagnostic_GPU",
                    "In_Situ_Diagnostic_Memory",
                    "In_Situ_Diagnostic_DataStorage",
                ],  # TODO: incorporate costs of extracting components when we have the option to repair them in-situ
            ),
            CEOption(
                "Recycle_Motherboard",
                base_value=value * health * 0.3,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(2.0 + np.random.randn() * (2.0 * 0.2)),
                Phi=0.5,
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
            ),
        ]
        self.value = value
        self.value_weight = f
        self._build_graph()

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates a dictionary of ground-truth health values for the product and its components."""
        state = {self.name: self.health_state.product_health}
        for component in self.components:
            state.update(component.generate_true_state(mode))
        return state


class Display(Product):
    """Leaf node component."""

    def __init__(self, health=1.0, value=100, f=1.0, name: str = "Display") -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        self.actions = [
            Action(
                "Bench_Diagnostic_Display",
                cost=2.0,
                time=2.0,
                prerequisites=["Power_On_Test"],
            ),
        ]
        self.ce_options = [
            CEOption(
                name="Reuse_Display",
                base_value=value * health * 0.95,
                phi_k=0.7 + (np.random.randn() * 0.2),
                E_k=0.0,
                Phi=1.0,
                fuzzy_params=(0.8, 0.9, 1.0, 1.0),
                prerequisites=["Bench_Diagnostic_Display"],
            ),
            CEOption(
                name="Repair_Display",
                base_value=value * health * 0.5,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)) * f,
                Phi=0.8,
                fuzzy_params=(0.4, 0.5, 0.8, 0.9),
                prerequisites=["Bench_Diagnostic_Display"],
            ),
            CEOption(
                name="Recycle_Display",
                base_value=value * health * 0.3,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(2.0 + np.random.randn() * (2.0 * 0.2)),
                Phi=0.5,
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
            ),
        ]
        self.value = value
        self.value_weight = f
        self._build_graph()

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates a dictionary of ground-truth health values for the product and its components."""
        return {self.name: self.health_state.product_health}


class Keyboard(Product):
    """Leaf node component."""

    def __init__(self, health=1.0, value=100, f=1.0, name: str = "Keyboard") -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        self.actions = [
            Action(
                "Bench_Diagnostic_Keyboard",
                cost=6.0,
                time=2.0,
                prerequisites=["Open_Chassis"],
            ),
        ]
        self.ce_options = [
            CEOption(
                name="Reuse_Keyboard",
                base_value=value * health * 0.95,
                phi_k=0.7 + (np.random.randn() * 0.2),
                E_k=0.0,
                Phi=1.0,
                fuzzy_params=(0.9, 0.95, 1.0, 1.0),
                prerequisites=["Bench_Diagnostic_Keyboard"],
            ),
            CEOption(
                name="Repair_Keyboard",
                base_value=value * health * 0.5,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)) * f,
                Phi=0.8,
                fuzzy_params=(0.5, 0.6, 0.8, 0.9),
                prerequisites=["Bench_Diagnostic_Keyboard"],
            ),
            CEOption(
                name="Recycle_Keyboard",
                base_value=value * health * 0.3,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(2.0 + np.random.randn() * (2.0 * 0.2)),
                Phi=0.5,
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
            ),
        ]
        self.value = value
        self.value_weight = f
        self._build_graph()

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates a dictionary of ground-truth health values for the product and its components."""
        return {self.name: self.health_state.product_health}


class Battery(Product):
    """Leaf node component."""

    def __init__(self, health=1.0, value=100, f=1.0, name: str = "Battery") -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        self.actions = [
            Action(
                "Bench_Diagnostic_Battery",
                cost=6.0,
                time=2.0,
                prerequisites=["Power_On_Test"],
            ),
        ]
        self.ce_options = [
            CEOption(
                name="Reuse_Battery",
                base_value=value * health * 0.95,
                phi_k=0.7 + (np.random.randn() * 0.2),
                E_k=0.0,
                Phi=1.0,
                fuzzy_params=(0.9, 0.95, 1.0, 1.0),
                prerequisites=["Bench_Diagnostic_Battery"],
            ),
            CEOption(
                name="Repair_Battery",
                base_value=value * health * 0.5,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)) * f,
                Phi=0.8,
                fuzzy_params=(0.5, 0.6, 0.8, 0.9),
                prerequisites=["Bench_Diagnostic_Battery"],
            ),
            CEOption(
                name="Recycle_Battery",
                base_value=value * health * 0.3,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(2.0 + np.random.randn() * (2.0 * 0.2)),
                Phi=0.5,
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
            ),
        ]
        self.value = value
        self.value_weight = f
        self._build_graph()

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates a dictionary of ground-truth health values for the product and its components."""
        return {self.name: self.health_state.product_health}


class Chassis(Product):
    """Leaf node component."""

    def __init__(self, health=1.0, value=50, f=1.0, name: str = "Chassis") -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        self.actions = [
            Action(
                "Bench_Diagnostic_Chassis",
                cost=4.0,
                time=2.0,
                prerequisites=["Open_Chassis", "Extract_Components"],
            ),
        ]
        self.ce_options = [
            CEOption(
                name="Reuse_Chassis",
                base_value=value * health * 0.95,
                phi_k=0.7 + (np.random.randn() * 0.2),
                E_k=0.0,
                Phi=1.0,
                fuzzy_params=(0.95, 0.98, 1.0, 1.0),
                prerequisites=["Bench_Diagnostic_Chassis"],
            ),
            CEOption(
                name="Repair_Chassis",
                base_value=value * health * 0.5,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)) * f,
                Phi=0.8,
                fuzzy_params=(0.5, 0.6, 0.9, 0.95),
                prerequisites=["Bench_Diagnostic_Chassis"],
            ),
            CEOption(
                name="Recycle_Chassis",
                base_value=value * health * 0.3,
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(2.0 + np.random.randn() * (2.0 * 0.2)),
                Phi=0.5,
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
            ),
        ]

        self.value = value
        self.value_weight = f
        self._build_graph()

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates a dictionary of ground-truth health values for the product and its components."""
        return {self.name: self.health_state.product_health}


class Laptop(Product):
    """Root assembly."""

    def __init__(
        self,
        health=1.0,
        value=1000,
        f=1.0,
        motherboard_health: float = 1.0,
        Display_health: float = 1.0,
        keyboard_health: float = 1.0,
        Battery_health: float = 1.0,
        Chassis_health: float = 1.0,
        name="Laptop",
    ) -> None:
        super().__init__(health_state=HealthState(product_health=health), name=name)
        self.components = [
            Motherboard(
                health=motherboard_health,
                f=0.47,  # weighted by emissions contribution, assumption that CE option value is correlated with emissions saving
                value=value * 0.47,
                name=f"{name}_Motherboard",
            ),
            Display(
                health=Display_health, f=0.3, value=value * 0.3, name=f"{name}_Display"
            ),
            Keyboard(
                health=keyboard_health,
                f=0.03,
                value=value * 0.03,
                name=f"{name}_Keyboard",
            ),
            Battery(
                health=Battery_health,
                f=0.06,
                value=value * 0.06,
                name=f"{name}_Battery",
            ),
            Chassis(
                health=Chassis_health,
                f=0.06,
                value=value * 0.06,
                name=f"{name}_Chassis",
            ),
        ]

        self.actions = [
            Action("Visual_Inspection", cost=5.0, time=2.0),
            Action(
                "Power_On_Test",
                cost=2.0,
                time=1.0,
                prerequisites=["Visual_Inspection"],
            ),
            Action(
                "In_Situ_Diagnostic_Display",
                cost=2.0,
                time=2.0,
                prerequisites=["Power_On_Test"],
            ),
            Action(
                "In_Situ_Diagnostic_Keyboard",
                cost=2.0,
                time=2.0,
                prerequisites=["Power_On_Test"],
            ),
            Action(
                "In_Situ_Diagnostic_Battery",
                cost=2.0,
                time=2.0,
                prerequisites=["Power_On_Test"],
            ),
            Action(
                "In_Situ_Diagnostic_Motherboard",
                cost=2.0,
                time=5.0,
                prerequisites=[
                    "Power_On_Test",
                    "In_Situ_Diagnostic_Display",
                    "In_Situ_Diagnostic_Keyboard",
                ],
            ),
            Action(
                "Open_Chassis",
                cost=15.0,
                time=10.0,
                prerequisites=["Power_On_Test"],
                is_disassembly=False,
            ),
            # create an action to make sure datastorage is wiped for reuse/repair options
            Action(
                "Wipe_DataStorage",
                cost=5.0,
                time=15.0,
                prerequisites=["Open_Chassis"],
                is_disassembly=False,
            ),
            Action(
                "Extract_Components",
                cost=30.0,
                time=20.0,
                prerequisites=["Open_Chassis"],
                is_disassembly=True,
            ),
        ]
        self.ce_options = [
            CEOption(
                "Reuse_Laptop",
                base_value=value * health * 1.0,  # this is recovery value, i.e beta
                fuzzy_params=(0.8, 0.9, 1.0, 1.0),
                phi_k=0.7
                + (np.random.randn() * 0.02),  # this is the pathway modifier, i.e. phi
                E_k=0.0,  # this is the emissions for reuse pathway, i.e. E
                Phi=1.0,  # this is the emissions ratio, its 1 because we recover all virgin material when reusing
                prerequisites=[
                    "Power_On_Test",
                    "In_Situ_Diagnostic_Display",
                    "In_Situ_Diagnostic_Keyboard",
                    "In_Situ_Diagnostic_Battery",
                    "In_Situ_Diagnostic_Motherboard",
                    "Wipe_DataStorage",
                ],
            ),
            CEOption(
                "Refurbish_Laptop",
                base_value=value * health * 0.6,
                fuzzy_params=(0.6, 0.7, 1.0, 1.0),
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)),
                Phi=0.8,
                prerequisites=[
                    "Open_Chassis",
                    "In_Situ_Diagnostic_Display",
                    "In_Situ_Diagnostic_Keyboard",
                    "In_Situ_Diagnostic_Battery",
                    "In_Situ_Diagnostic_Motherboard",
                    "Wipe_DataStorage",
                ],
            ),
            CEOption(
                "Repair_Laptop",
                base_value=value * health * 0.5,
                fuzzy_params=(0.4, 0.5, 0.7, 0.8),
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(21.0 + np.random.randn() * (21.0 * 0.2)),
                Phi=0.8,
                prerequisites=[
                    "Open_Chassis",
                    "Wipe_DataStorage",
                ],
            ),
            CEOption(
                "Recycle_Laptop",
                base_value=value * health * 0.3,
                fuzzy_params=(0.0, 0.0, 1.0, 1.0),
                phi_k=1.0 + (np.random.randn() * 0.02),
                E_k=(2.0 + np.random.randn() * (2.0 * 0.2)),
                Phi=0.5,
                prerequisites=[
                    "Wipe_DataStorage",
                ],
            ),
        ]
        self.value = value
        self.value_weight = f

        self._build_graph()

    def generate_true_state(self, mode="uniform") -> Dict[str, float]:
        """Generates ground-truth health for the laptop and its components."""
        state = {}
        component_healths = []

        # Generate health for each component based on mode
        if mode == "uniform":
            comp_healths = {
                "Motherboard": 0.8,
                "Display": 0.8,
                "Keyboard": 0.8,
                "Battery": 0.8,
                "Chassis": 0.8,
                "CPU": 0.8,
                "GPU": 0.8,
                "Memory": 0.8,
                "DataStorage": 0.8,
            }
        elif mode == "moderate":
            comp_healths = {
                "Motherboard": 0.9,
                "Display": 0.7,
                "Keyboard": 0.8,
                "Battery": 0.9,
                "Chassis": 0.7,
                "CPU": 0.9,
                "GPU": 0.7,
                "Memory": 0.8,
                "DataStorage": 0.9,
            }
        elif mode == "extreme":
            comp_healths = {
                "Motherboard": 1.0,
                "Display": 0.4,
                "Keyboard": 1.0,
                "Battery": 1.0,
                "Chassis": 0.4,
                "CPU": 1.0,
                "GPU": 0.4,
                "Memory": 1.0,
                "DataStorage": 1.0,
            }
        # elif mode =="random":
        #     comp_healths = {
        #         "Motherboard": np.random.uniform(0.4, 1.0),
        #         "Display": np.random.uniform(0.4, 1.0),
        #         "Keyboard": np.random.uniform(0.4, 1.0),
        #         "Battery": np.random.uniform(0.4, 1.0),
        #         "Chassis": np.random.uniform(0.4, 1.0),
        #         "CPU": np.random.uniform(0.4, 1.0),
        #         "GPU": np.random.uniform(0.4, 1.0),
        #         "Memory": np.random.uniform(0.4, 1.0),
        #         "DataStorage": np.random.uniform(0.4, 1.0),
        #     }
        else:  # Random
            comp_healths = {
                comp.name.split("_")[-1]: np.random.uniform(0.2, 1.0)
                for comp in self.components
            }
            comp_healths["CPU"] = np.random.uniform(0.2, 1.0)
            comp_healths["GPU"] = np.random.uniform(0.2, 1.0)
            comp_healths["Memory"] = np.random.uniform(0.2, 1.0)
            comp_healths["DataStorage"] = np.random.uniform(0.2, 1.0)

        # Set health for each component and add to state
        for comp in self.components:
            comp_name_key = comp.name.split("_")[-1]
            health = comp_healths.get(comp_name_key, 1.0)
            comp.health_state.product_health = health
            state.update(comp.generate_true_state())
            component_healths.append(health)

            if isinstance(comp, Motherboard):
                for mb_comp in comp.components:
                    mb_comp_name_key = mb_comp.name.split("_")[-1]
                    mb_health = comp_healths.get(mb_comp_name_key, 1.0)
                    mb_comp.health_state.product_health = mb_health
                    state.update(mb_comp.generate_true_state())

        # 3. Build Laptop from average of component healths (simple heuristic for overall laptop health)
        laptop_health = np.random.uniform(0.2, 1.0)
        state[self.name] = laptop_health

        return state


if __name__ == "__main__":
    laptop = Laptop()
    for mode in ["uniform", "moderate", "extreme", "random"]:
        true_state = laptop.generate_true_state(mode=mode)
        print(f"Mode: {mode}")
        for comp, health in true_state.items():
            print(f"  {comp}: {health:.2f}")
