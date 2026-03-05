import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from util import Sensor, draw_labeled_multigraph
from battery import Battery
from model import StateAugmentedGraph, Solver, Node


class Simulation:
    def __init__(self):
        # 1. Instantiate the physical product template (Root Product)
        print("Initializing Base Battery Template...")
        self.base_product = Battery(name="Battery_Pack")

        # 2. Build the unrolled Decision Graph (State-Augmented Graph)
        print("Building State-Augmented Graph (SAG)...")
        self.sag = StateAugmentedGraph(self.base_product)
        print(
            f"  -> SAG built with {len(self.sag.graph.nodes)} nodes and {len(self.sag.graph.edges)} edges."
        )

        self.solver = Solver()

    def inspect_dag(self):
        """Builds and plots the base Physical DAG (No History/State Augmentation)."""
        print("Rendering Base Physical DAG...")
        dag = nx.MultiDiGraph()

        # Simple recursive traversal to build the pure physical graph
        def traverse(prod):
            dag.add_node(prod)
            for act in prod.actions:
                if act.is_disassembly:
                    for comp in prod.components:
                        dag.add_node(comp)
                        dag.add_edge(prod, comp, action=act.name)
                        traverse(comp)
                else:
                    # Prep actions loop back to the same physical product in a pure DAG
                    dag.add_edge(prod, prod, action=act.name)

            for ce in prod.ce_options:
                term_node = f"TERMINAL_{ce.name}_{prod.name}"
                dag.add_node(term_node)
                dag.add_edge(prod, term_node, action=ce.name)

        traverse(self.base_product)

        fig, ax = plt.subplots(figsize=(14, 10))
        draw_labeled_multigraph(dag, attr_name="action", ax=ax)
        fig.suptitle("Base Product DAG (Physical Connections Only)", fontsize=16)

        # Show but DO NOT block the script
        plt.show(block=False)
        plt.pause(0.1)

    def inspect_graph(self):
        """Plots the State-Augmented Graph using NetworkX."""
        print("Rendering State-Augmented Graph...")
        fig, ax = plt.subplots(figsize=(16, 12))

        draw_labeled_multigraph(self.sag.graph, attr_name="action", ax=ax)
        fig.suptitle(
            "State-Augmented Graph (SAG) - Unrolled Markov Decision Space", fontsize=16
        )

        # Show but DO NOT block the script
        plt.show(block=False)
        plt.pause(0.1)

    def generate_true_state(self, mode="uniform"):
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
        state["Battery_Pack"] = pack_health

        return state

    def run_single_trace(self, noise_level=0.1):
        """Runs one episode and prints the exact path taken for debugging/demo."""
        print("\n" + "=" * 70)
        print(f"RUNNING SINGLE TRACE EPISODE (Noise Std: {noise_level})")
        print("=" * 70)

        sensor = Sensor(noise_std=noise_level)
        true_state = self.generate_true_state()

        # Generate Noisy Observations
        observations = {k: sensor.read(v) for k, v in true_state.items()}

        print("GROUND TRUTH VS OBSERVATIONS:")
        for k in true_state.keys():
            print(
                f"  {k}: True={true_state[k]:.3f} | Observed={observations.get(k, 0.5):.3f}"
            )

        print("\nSOLVING POLICY...")
        policy = self.solver.solve(self.sag.root, observations)

        print("\nEXECUTING POLICY (Walking the tree):")
        actual_profit = self.calculate_real_world_profit(
            self.sag.root, policy, true_state, verbose=True
        )

        print(f"\nFINAL EPISODE PROFIT: £{actual_profit:.2f}")
        print("=" * 70 + "\n")

    def run_experiment(self, n_episodes=1000, noise_level=0.1, state_mode="uniform"):
        """Runs a Monte Carlo batch to evaluate average real-world profit."""
        sensor = Sensor(noise_std=noise_level)
        results = []

        for _ in range(n_episodes):
            true_state = self.generate_true_state(mode=state_mode)

            # 1. Generate Noisy Observations
            observations = {k: sensor.read(v) for k, v in true_state.items()}

            # 2. Solver plans based on NOISE
            policy = self.solver.solve(self.sag.root, observations)

            # 3. Evaluate Outcome based on TRUTH
            # The solver makes its decision path based on what it *thought* the health was.
            # But the reward we actually collect depends on the *true* health of the item.
            actual_profit = self.calculate_real_world_profit(
                self.sag.root, policy, true_state, verbose=False
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
        }

    # def plot_fuzzy_reward_curves(self):
    #     """Visualizes the fuzzy reward curves and the resulting decision envelope."""
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from model import CEOption

    #     # 1. Define the CE Options for a Battery Pack (Matching battery.py)
    #     ce_options = [
    #         CEOption(
    #             "Reuse_EV_Pack", base_value=800.0, fuzziness_k=20.0, threshold=0.9
    #         ),
    #         CEOption(
    #             "Repurpose_Storage", base_value=400.0, fuzziness_k=10.0, threshold=0.7
    #         ),
    #         CEOption(
    #             "Recycle_Pack_Direct", base_value=50.0, fuzziness_k=1.0, threshold=0.0
    #         ),
    #     ]

    #     # 2. Generate an array of possible health observations
    #     health_x = np.linspace(0.0, 1.0, 500)

    #     # 3. Calculate the expected reward for each option across all healths
    #     rewards_y = {opt.name: [] for opt in ce_options}
    #     envelope_y = []  # The maximum value at any given health (The MDP's choice)

    #     for h in health_x:
    #         max_val = -float("inf")
    #         for opt in ce_options:
    #             r = opt.get_expected_reward(h)
    #             rewards_y[opt.name].append(r)
    #             if r > max_val:
    #                 max_val = r
    #         envelope_y.append(max_val)

    #     # 4. Plotting
    #     fig, ax = plt.subplots(figsize=(10, 6))

    #     colors = {
    #         "Reuse_EV_Pack": "green",
    #         "Repurpose_Storage": "orange",
    #         "Recycle_Pack_Direct": "red",
    #     }

    #     # Plot individual fuzzy curves
    #     for opt_name, y_vals in rewards_y.items():
    #         ax.plot(
    #             health_x,
    #             y_vals,
    #             label=f"{opt_name} Curve",
    #             color=colors[opt_name],
    #             linestyle="--",
    #             alpha=0.7,
    #             linewidth=2,
    #         )

    #     # Plot the actual MDP Decision Envelope
    #     ax.plot(
    #         health_x,
    #         envelope_y,
    #         label="MDP Decision Envelope (Max)",
    #         color="black",
    #         linewidth=3,
    #     )

    #     # Highlight the decision crossover points
    #     ax.fill_between(health_x, 0, envelope_y, color="gray", alpha=0.1)

    #     ax.set_title(
    #         "Fuzzy Reward Curves and MDP Decision Envelope (Battery Pack)", fontsize=14
    #     )
    #     ax.set_xlabel("Observed State Health", fontsize=12)
    #     ax.set_ylabel("Expected Value / Reward (£)", fontsize=12)
    #     ax.grid(True, linestyle="--", alpha=0.6)
    #     ax.legend(loc="upper left")

    #     # Annotate
    #     ax.annotate(
    #         "Threshold: High Risk Penalty",
    #         xy=(0.85, 400),
    #         xytext=(0.6, 600),
    #         arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
    #     )

    #     plt.tight_layout()
    #     plt.show(block=False)
    #     plt.pause(0.1)

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

        colors = {
            "Reuse_EV_Pack": "forestgreen",
            "Repurpose_Storage": "darkorange",
            "Recycle_Pack_Direct": "crimson",
        }

        for opt_name, y_vals in rewards_y.items():
            ax.plot(
                health_x,
                y_vals,
                label=f"{opt_name} (Fuzzy Yield)",
                color=colors.get(opt_name, "blue"),
                linestyle="--",
                alpha=0.8,
                linewidth=2,
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
            "Trapezoidal Fuzzy Reward Curves & Value Envelope",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Observed True Health", fontsize=12)
        ax.set_ylabel("Expected Financial Reward (£)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left")

        # Add visual markers for the "Safe Zones" (Dynamically placed based on max reward)
        max_reward = max(envelope_y) if envelope_y else 800

        ax.axvline(x=0.4, color="black", linestyle="--")
        ax.axvline(x=0.7, color="black", linestyle="--")
        ax.axvline(x=0.8, color="black", linestyle="--")
        ax.axvline(x=1.0, color="black", linestyle="--")
        # ax.annotate(
        #     "Core Safe Zone\n(Full Value)",
        #     xy=(0.95, max_reward),
        #     xytext=(0.85, max_reward * 0.75),
        #     arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
        # )
        # ax.annotate(
        #     "Transition Zone\n(Risk Penalty)",
        #     xy=(0.85, max_reward * 0.5),
        #     xytext=(0.7, max_reward * 0.6),
        #     arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
        # )

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
