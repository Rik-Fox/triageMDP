import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from util import Sensor, draw_labeled_multigraph, Product
from battery import Battery
from model import StateAugmentedDisassemblyGraph, Solver, Node


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

    def run_single_trace(self, noise_level=0.1):
        """Runs one episode and prints the exact path taken for debugging/demo."""
        print("\n" + "=" * 70)
        print(f"RUNNING SINGLE TRACE EPISODE (Noise Std: {noise_level})")
        print("=" * 70)

        sensor = Sensor(noise_std=noise_level)
        true_state = self.base_product.generate_true_state()

        # Generate Noisy Observations
        observations = {k: sensor.read(v) for k, v in true_state.items()}

        print("GROUND TRUTH VS OBSERVATIONS:")
        for k in true_state.keys():
            print(
                f"  {k}: True={true_state[k]:.3f} | Observed={observations.get(k, 0.5):.3f}"
            )

        print("\nSOLVING POLICY...")
        policy = self.solver.solve(
            self.sadg.state_map[self.sadg.root_key], observations
        )

        print("\nEXECUTING POLICY (Walking the tree):")
        actual_profit = self.calculate_real_world_profit(
            self.sadg.state_map[self.sadg.root_key], policy, true_state, verbose=True
        )

        print(f"\nFINAL EPISODE PROFIT: £{actual_profit:.2f}")
        print("=" * 70 + "\n")

    def run_experiment(self, n_episodes=1000, noise_level=0.1, state_mode="uniform"):
        """Runs a Monte Carlo batch to evaluate average real-world profit."""
        sensor = Sensor(noise_std=noise_level)
        results = []

        for _ in range(n_episodes):

            true_state = self.base_product.generate_true_state(mode=state_mode)

            observations = {k: sensor.read(v) for k, v in true_state.items()}
            policy = self.solver.solve(
                self.sadg.state_map[self.sadg.root_key], observations
            )

            actual_profit = self.calculate_real_world_profit(
                self.sadg.state_map[self.sadg.root_key],
                policy,
                true_state,
                verbose=False,
            )
            results.append(actual_profit)

        return self.analyse_results(results)

    def calculate_real_world_profit(
        self,
        current_node: Node,
        policy: dict,
        true_state: dict,
        verbose=False,
        indent="",
    ) -> float:
        """
        Recursively walks the policy graph, accumulating costs and collecting
        terminal fuzzy rewards evaluated against the TRUE state.
        """
        decision = policy.get(current_node.id)

        if not decision:
            if verbose:
                print(
                    f"{indent}Reached node {current_node.id} with no policy. Returning 0."
                )
            return 0.0

        if decision.startswith("CE:"):
            ce_name = decision.split("CE: ")[1]
            if verbose:
                print(f"{indent}-> DECISION: Commit to CE Option [{ce_name}]")

            # Find the CE option object
            ce_opt = next(
                (opt for opt in current_node.ce_options if opt.name == ce_name), None
            )
            if ce_opt:
                # Evaluate using TRUE health, not noisy observation!
                true_h = true_state.get(current_node.product.name, 0.5)
                reward = ce_opt.get_expected_reward(true_h)
                if verbose:
                    print(
                        f"{indent}   (True Health: {true_h:.3f} -> Fuzzy Reward: £{reward:.2f})"
                    )
                return reward
            return 0.0

        elif decision.startswith("Action:"):
            act_name = decision.split("Action: ")[1]
            if verbose:
                print(f"{indent}-> DECISION: Perform Action [{act_name}]")

            # Find the action and the children it leads to
            for action, child_nodes in current_node.children:
                if action.name == act_name:
                    cost = action.cost
                    if verbose:
                        print(f"{indent}   (Action Cost: -£{cost:.2f})")

                    # Sum the real profit of all resulting children (handles splitting into multiple modules)
                    children_profit = 0.0
                    for child in child_nodes:
                        children_profit += self.calculate_real_world_profit(
                            child, policy, true_state, verbose, indent + "    "
                        )

                    return -cost + children_profit
            return 0.0

        return 0.0

    def analyse_results(self, results):
        """Calculates true statistical metrics including Standard Deviation."""
        return {
            "average_profit": np.mean(results),
            "std_profit": np.std(results),  # Real variance calculation
            "min_profit": np.min(results),
            "max_profit": np.max(results),
            "total_episodes": len(results),
            "raw_profits": results,  # For potential further analysis
        }

    def plot_fuzzy_reward_curves(self):
        """Visualizes the Trapezoidal fuzzy reward curves and the resulting decision envelope."""
        print("\nPlotting Trapezoidal Fuzzy Reward Curves...")

        # 1. Fetch CE Options dynamically from the base pack
        ce_options = self.base_product.ce_options

        # 2. Generate health observations
        health_x = np.linspace(0.0, 1.0, 500)

        rewards_y = {opt.name: [] for opt in ce_options}
        envelope_y = []

        for h in health_x:
            max_val = -float("inf")
            for opt in ce_options:
                r = opt.get_expected_reward(h)
                rewards_y[opt.name].append(r)
                if r > max_val:
                    max_val = r
            envelope_y.append(max_val)

        # 3. Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(ce_options)))

        curve_fill_alpha = 0.2

        for i, (opt_name, y_vals) in enumerate(rewards_y.items()):
            curve_color = colors[i]

            # Plot the dashed boundary line
            ax.plot(
                health_x,
                y_vals,
                label=f"{opt_name} (Fuzzy Yield)",
                color=curve_color,
                linestyle="--",
                alpha=0.8,
                linewidth=2,
            )

            # Fill the area under the individual trapezoid curve
            ax.fill_between(
                health_x,
                0,  # Base of the fill (y=0)
                y_vals,  # Top of the fill (the curve)
                color=curve_color,
                alpha=curve_fill_alpha,
                linewidth=0,  # Removes the stroke from the fill itself
            )

        # Plot the actual MDP Decision Envelope (The max function)
        ax.plot(
            health_x,
            envelope_y,
            label="MDP Expected Value Envelope",
            color="black",
            linewidth=3,
        )
        ax.fill_between(health_x, 0, envelope_y, color="gray", alpha=0.15)

        ax.set_title(
            f"Fuzzy Reward Curves for a {self.base_product.name.split('_')[-1]}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("True Health", fontsize=12)
        ax.set_ylabel("Expected Reward", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
