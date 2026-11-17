import matplotlib.pyplot as plt
from util import draw_labeled_multigraph, Product
from model import StateAugmentedDisassemblyGraph, Solver


class Simulation:
    def __init__(self, product: Product):
        # 1. Instantiate the physical product template (Root Product)
        print(f"Initializing Base {product.name} Template...")
        self.base_product = product

        # 2. Build the unrolled Decision Graph (State-Augmented Graph)
        print("Building State-Augmented Graph (sadg)...")
        self.sadg = StateAugmentedDisassemblyGraph(self.base_product)
        print(
            f"  -> sadg built with {len(self.sadg.graph.nodes)} nodes and {len(self.sadg.graph.edges)} edges."
        )

        self.solver = Solver()

    def inspect_dag(self):
        """plots the base Physical DAG (No History/State Augmentation)."""

        fig, ax = plt.subplots(figsize=(14, 10))
        draw_labeled_multigraph(self.base_product.graph, attr_name="action", ax=ax)
        fig.suptitle("Base Product DAG (Physical Connections Only)", fontsize=16)

        # Show but DO NOT block the script
        plt.show(block=False)
        plt.pause(0.1)

    def inspect_graph(self):
        """Plots the State-Augmented Graph using NetworkX."""
        print("Rendering State-Augmented Graph...")
        fig, ax = plt.subplots(figsize=(16, 12))

        draw_labeled_multigraph(self.sadg.graph, attr_name="action", ax=ax)
        fig.suptitle(
            "State-Augmented Graph (sadg) - Unrolled Markov Decision Space", fontsize=16
        )

        # Show but DO NOT block the script
        plt.show(block=False)
        plt.pause(0.1)
