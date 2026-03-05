# from typing import Dict, List, Tuple
# import itertools as it
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# from dataclasses import dataclass


# @dataclass
# class HealthState:
#     """The 'God Mode' truth of the product."""

#     product_health: float  # e.g., 0.85 for 85% healthy
#     component_healths: dict[
#         str, float
#     ]  # e.g., {"Module_A": 0.85, "Module_B": 0.4} -1 if missing

#     def adjust_health(self, factor: float) -> None:
#         """Adjusts the health of a specific component by a factor."""
#         product_health = np.clip(self.product_health * factor, 0.0, 1.0)
#         self.product_health = product_health
#         # Adjust individual component healths - this is a naive implementation;
#         # where components are adjusted equally by the same factor
#         for comp in self.component_healths:
#             if self.component_healths[comp] >= 0.0:
#                 self.component_healths[comp] = np.clip(
#                     self.component_healths[comp] * factor, 0.0, 1.0
#                 )


# class Sensor:
#     """Simulates the noise in diagnostic tools."""

#     def __init__(self, noise_std: float) -> None:
#         self.noise_std = noise_std

#     def read(self, true_health: float) -> float:
#         # Simple Gaussian noise, clipped to [0, 1]
#         reading = true_health + np.random.normal(0, self.noise_std)
#         return np.clip(reading, 0.0, 1.0)


# class Action:
#     """Represents an action that can be taken on a product."""

#     def __init__(
#         self, name: str, cost: float, time: float, resources: list = []
#     ) -> None:
#         self.name = name
#         self.cost = cost
#         self.time = time
#         self.resources = resources


# class Product:
#     """Represents a product with multiple components."""

#     def __init__(
#         self,
#         health_state: HealthState,
#         ID: int = np.random.randint(1e2),
#         name: str = "GenericProduct",
#     ) -> None:
#         self.ID = ID
#         # self.name = f"{name}_{ID}"
#         self.name = f"{name}"
#         self.health_state = health_state
#         self.component_list = list(self.health_state.component_healths.keys())

#         self.components = []
#         self.actions = []

#         self.graph = self._build_graph()

#         self.children = []  # List of resulting products after disassembly actions

#     def _build_graph(self) -> Dict:
#         """Builds the leaf node for the DAG."""

#         G = nx.MultiDiGraph()

#         G.add_node(self)

#         return G

#     def get_graph(self) -> Dict[str, List[Tuple[str, List[str], float, float]]]:
#         """Returns the disassembly graph."""
#         return self.graph

#     def get_all_products(self) -> List[str]:
#         """Returns a list of all product names in the graph."""
#         return list(self.graph.keys())


# class ProductGenerator:
#     """Factory for creating random product instances."""

#     def generate_batch(self, size: int) -> list[Product]:
#         # Logic to generate products with varying defects
#         products = []
#         for _ in range(size):
#             healths = {
#                 "Module_A": np.random.uniform(low=0.0, high=1.0),
#                 "Module_B": np.random.uniform(low=0.0, high=1.0),
#                 "Module_C": np.random.uniform(low=0.0, high=1.0),
#             }
#             products.append(
#                 Product(health_state=HealthState(component_healths=healths))
#             )


# def draw_labeled_multigraph(G, attr_name, ax=None):
#     """
#     Length of connectionstyle must be at least that of a maximum number of edges
#     between pair of nodes. This number is maximum one-sided connections
#     for directed graph and maximum total connections for undirected graph.
#     """
#     # Works with arc3 and angle3 connectionstyles
#     # connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
#     connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 8)]

#     # pos = nx.shell_layout(G)
#     pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="")  # "twopi" "dot" "circo"
#     # pos = nx.spring_layout(G, seed=173420, method="force")
#     nx.draw_networkx_nodes(G, pos, ax=ax)
#     nx.draw_networkx_labels(
#         G, pos, labels={node: node.name for node in G.nodes}, font_size=20, ax=ax
#     )
#     nx.draw_networkx_edges(
#         G,
#         pos,
#         edge_color="grey",
#         # connectionstyle=connectionstyle,
#         ax=ax,
#     )

#     labels = {
#         tuple(edge): f"{attr_name}={attrs[attr_name]}"
#         for *edge, attrs in G.edges(keys=True, data=True)
#     }
#     nx.draw_networkx_edge_labels(
#         G=G,
#         pos=pos,
#         edge_labels=labels,
#         # connectionstyle=connectionstyle,
#         label_pos=0.3,
#         font_color="blue",
#         bbox={"alpha": 0},
#         ax=ax,
#     )

from typing import Dict, List, Tuple, Optional
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass, field


@dataclass
class HealthState:
    """The 'God Mode' truth of the product."""

    product_health: float  # e.g., 0.85 for 85% healthy
    component_healths: dict[str, float] = field(default_factory=dict)

    def adjust_health(self, factor: float) -> None:
        """Adjusts the health of a specific component by a factor."""
        self.product_health = np.clip(self.product_health * factor, 0.0, 1.0)
        # Naive implementation: adjust all sub-components equally
        for comp in self.component_healths:
            if self.component_healths[comp] >= 0.0:
                self.component_healths[comp] = np.clip(
                    self.component_healths[comp] * factor, 0.0, 1.0
                )

    def __repr__(self):
        return f"Health(Overall={self.product_health:.2f}, Components={len(self.component_healths)})"


class Sensor:
    """Simulates the noise in diagnostic tools."""

    def __init__(self, noise_std: float) -> None:
        self.noise_std = noise_std

    def read(self, true_health: float) -> float:
        reading = true_health + np.random.normal(0, self.noise_std)
        return float(np.clip(reading, 0.0, 1.0))


class Action:
    """Represents a generic action (Disassembly, Prep, or CE Option)."""

    def __init__(
        self,
        name: str,
        cost: float,
        time: float,
        resources: list = None,
        prerequisites: list = None,
        is_disassembly: bool = False,
    ) -> None:
        self.name = name
        self.cost = cost
        self.time = time
        self.resources = resources if resources else []
        self.prerequisites = prerequisites if prerequisites else []
        self.is_disassembly = is_disassembly

    def __repr__(self):
        return f"{self.name}(Cost={self.cost}, Prereqs={self.prerequisites})"


class Product:
    """Represents a physical product/component."""

    def __init__(
        self,
        health_state: HealthState,
        ID: Optional[int] = None,
        name: str = "GenericProduct",
    ) -> None:
        self.ID = ID if ID is not None else np.random.randint(1e5)
        self.name = f"{name}"  # Simplified naming for clarity
        self.health_state = health_state

        # Physical hierarchy
        self.components: List["Product"] = []

        # Available actions on this specific node
        self.actions: List[Action] = []  # Disassembly Actions
        self.ce_options: List[Action] = []  # Terminal Options (Reuse/Recycle)

        # Graph placeholder
        self.graph = nx.MultiDiGraph()

    def get_components(self) -> List["Product"]:
        return self.components

    def _build_graph(self) -> nx.MultiDiGraph:
        """Helper to verify physical connections (visual mostly)."""
        G = nx.MultiDiGraph()
        G.add_node(self)
        return G


class ProductGenerator:
    """Factory for creating random product instances."""

    def generate_batch(self, size: int) -> list[Product]:
        products = []
        for i in range(size):
            # Example generation logic
            h_state = HealthState(
                product_health=np.random.uniform(0.5, 1.0), component_healths={"A": 0.9}
            )
            products.append(Product(health_state=h_state, ID=i))
        return products


def draw_labeled_multigraph(G, attr_name="action", ax=None):
    """Visualization helper using NetworkX with abbreviated histories and Y-staggering."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 12))

    # ---------------------------------------------------------
    # 1. BUILD LEGEND MAPPING FOR HISTORIES (e_x notation)
    # ---------------------------------------------------------
    unique_actions = set()
    for n in G.nodes:
        name_str = n.name if hasattr(n, "name") else str(n)
        if "_[" in name_str:
            hist_part = name_str.split("_[")[1].replace("]", "")
            if hist_part:
                for act in hist_part.split(","):
                    unique_actions.add(act.strip())

    # Create mapping e.g., {'High_Voltage_Isolation': 'e1', ...}
    action_map = {act: f"e{i+1}" for i, act in enumerate(sorted(unique_actions))}

    # ---------------------------------------------------------
    # 2. LAYOUT CALCULATION (With PyGraphviz DOT Bug Fix)
    # ---------------------------------------------------------
    try:
        mapping = {node: i for i, node in enumerate(G.nodes())}
        reverse_mapping = {i: node for node, i in mapping.items()}
        G_safe = nx.relabel_nodes(G, mapping)

        pos_safe = nx.nx_agraph.graphviz_layout(
            G_safe, prog="dot", args="-Gnodesep=1.0 -Granksep=2.0"
        )

        X_STRETCH = 2.0
        Y_STRETCH = 1.5
        pos = {
            reverse_mapping[i]: (x * X_STRETCH, y * Y_STRETCH)
            for i, (x, y) in pos_safe.items()
        }

    except ImportError:
        pos = nx.shell_layout(G)
        pos = {n: (x * 1000, y * 1000) for n, (x, y) in pos.items()}

    # ---------------------------------------------------------
    # 3. Y-STAGGERING FOR SAME-LEVEL NODES
    # ---------------------------------------------------------
    # Group nodes by approximate Y level (within 50 units)
    y_levels = {}
    for node, (x, y) in pos.items():
        placed = False
        for level_y in y_levels:
            if abs(y - level_y) < 50:
                y_levels[level_y].append(node)
                placed = True
                break
        if not placed:
            y_levels[y] = [node]

    # Apply alternating vertical stagger to prevent parallel boxes from colliding
    for level_nodes in y_levels.values():
        if len(level_nodes) > 1:
            level_nodes.sort(key=lambda n: pos[n][0])  # Sort horizontally
            for i, node in enumerate(level_nodes):
                x, original_y = pos[node]
                offset = 30 if i % 2 == 0 else -30  # Stagger up/down
                pos[node] = (x, original_y + offset)

    # ---------------------------------------------------------
    # 4. OPTIONAL OVERRIDE: Stack terminal nodes to the side
    # ---------------------------------------------------------
    temp_terminals = [
        n
        for n in G.nodes
        if (hasattr(n, "name") and n.name.startswith("TERMINAL_"))
        or str(n).startswith("TERMINAL_")
    ]
    temp_normals = [n for n in G.nodes if n not in temp_terminals]

    for u in temp_normals:
        term_children = [v for _, v in G.out_edges(u) if v in temp_terminals]
        if term_children:
            x_parent, y_parent = pos[u]
            for i, v in enumerate(term_children):
                pos[v] = (x_parent + 400, y_parent + ((i + 1) * 45))

    # ---------------------------------------------------------
    # 5. FORMAT NODE LABELS & IDENTIFY TERMINALS
    # ---------------------------------------------------------
    node_labels = {}
    terminal_nodes = []
    normal_nodes = []

    for n in G.nodes:
        name_str = n.name if hasattr(n, "name") else str(n)

        if name_str.startswith("TERMINAL_"):
            terminal_nodes.append(n)
            in_edges = list(G.in_edges(n, data=True))
            if in_edges:
                act_name = in_edges[0][2].get(attr_name, "CE Option")
                node_labels[n] = f"★ {act_name}"
            else:
                node_labels[n] = "★ CE Option"
        else:
            normal_nodes.append(n)
            if "_[]" in name_str:
                node_labels[n] = name_str.replace("_[]", "")
            elif "_[" in name_str:
                base_name, hist_part = name_str.split("_[")
                hist_part = hist_part.replace("]", "")
                # Swap the long action names for their e_x shorthand
                short_hist = ", ".join(
                    [action_map.get(act.strip(), act) for act in hist_part.split(",")]
                )
                node_labels[n] = f"{base_name}\n[{short_hist}]"
            else:
                node_labels[n] = name_str

    # ---------------------------------------------------------
    # 6. RENDER GRAPH ELEMENTS
    # ---------------------------------------------------------
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        arrows=True,
        node_size=2000,
        connectionstyle="arc3,rad=0.0",
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: node_labels[n] for n in normal_nodes},
        ax=ax,
        font_size=9,
        bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="steelblue", lw=1.5),
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: node_labels[n] for n in terminal_nodes},
        ax=ax,
        font_size=9,
        font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="papayawhip", ec="darkorange", lw=1.5),
    )

    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if v in terminal_nodes:
            continue
        if attr_name in data:
            label = f"{data[attr_name]}"
            if "cost" in data and data["cost"] > 0:
                label += f"\n(-£{data['cost']})"
            edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax, font_size=8, font_color="darkred"
    )

    # ---------------------------------------------------------
    # 7. RENDER LEGEND
    # ---------------------------------------------------------
    if action_map:
        legend_text = "History Legend ($\\tau$):\n" + "-" * 30 + "\n"
        legend_text += "\n".join([f"{v}: {k}" for k, v in action_map.items()])
        props = dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.9,
            edgecolor="steelblue",
            lw=1.5,
        )
        ax.text(
            0.02,
            0.98,
            legend_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
            family="monospace",
        )


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Core Utilities and Sensor Noise")
    print("-" * 60)
    print("SCENARIO:")
    print("  We create a 'God Mode' product with a true health of 0.85.")
    print("  We then pass this through a Sensor with a 5% standard deviation.")
    print("EXPECTED OUTCOME:")
    print("  The sensor reading should deviate slightly from 0.85, proving")
    print("  our Monte Carlo inputs will have realistic diagnostic uncertainty.")
    print("-" * 60)

    # 1. Setup Ground Truth
    true_health = 0.85
    hs = HealthState(product_health=true_health, component_healths={"CompA": 0.8})
    p = Product(hs, name="Test_Inverter")

    # 2. Setup Sensor
    sensor = Sensor(noise_std=0.05)

    # 3. Execution
    print(f"INPUT   : {p.name} created. True Health = {true_health:.3f}")

    readings = [sensor.read(true_health) for _ in range(3)]
    for i, reading in enumerate(readings):
        error = abs(true_health - reading)
        print(f"OUTPUT {i+1}: Sensor reads {reading:.3f} (Error: {error:.3f})")

    print("=" * 60 + "\n")
