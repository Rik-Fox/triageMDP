import os
import pandas as pd
import json
from sim import Simulation
import matplotlib.pyplot as plt
import numpy as np
from util import Product
from model import StateAugmentedDisassemblyGraph, Solver, Node


def run_noise_sensitivity_experiment(
    sim: Simulation,
    noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    episodes_per_level=500,
):
    """Runs a sweep over different sensor noise levels and plots the result."""
    print(
        f"Starting Noise Sensitivity Monte Carlo Experiment for {sim.base_product.name}..."
    )

    results = {"avg": [], "std": []}

    fig, ax = plt.subplots(figsize=(10, 6))

    for noise in noise_levels:
        print(f"  Running {episodes_per_level} episodes at noise std {noise:.2f}...")
        metrics = sim.run_experiment(n_episodes=episodes_per_level, noise_level=noise)
        results["avg"].append(metrics["average_profit"])
        results["std"].append(metrics["std_profit"])
        # print(f"Avg Profit for noise {noise}: £{metrics['average_profit']:.2f}")
        # print(f"Std Profit for noise {noise}: {metrics['std_profit']:.2f}")

        # Ensure your Simulation class returns the raw array of all episode profits
        raw_profits = metrics["raw_profits"]

        # Plot individual data points with high transparency to show density
        ax.scatter(
            [noise] * episodes_per_level,
            raw_profits,
            color="b",
            alpha=0.01,  # extremely low alpha due to 5000 points
            s=10,  # small point size
            edgecolors="none",
            zorder=1,
        )

    avg_profits = np.array(results["avg"])
    std_profits = np.array(results["std"])

    ax.plot(
        noise_levels,
        avg_profits,
        marker="o",
        linestyle="-",
        color="b",
        linewidth=2,
        label="Average Profit",
    )

    # Real Confidence Interval / Standard Deviation shading
    ax.fill_between(
        noise_levels,
        avg_profits - std_profits,
        avg_profits + std_profits,
        color="b",
        alpha=0.15,
        label="±1 Standard Deviation",
    )

    ax.set_title(
        f"Impact of Diagnostic Sensor Noise on Triage Profitability for {sim.base_product.name}",
        fontsize=14,
    )
    ax.set_xlabel("Sensor Noise (Standard Deviation)", fontsize=12)
    ax.set_ylabel("Average Realized Profit (£)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")

    # Add an annotation explaining the shaded region
    # ax.annotate(
    #     "Variance (shaded region) widens\nas noise unpredictability increases.",
    #     xy=(noise_levels[-2], avg_profits[-2] + std_profits[-2]),
    #     xytext=(noise_levels[-3], avg_profits[-3] + std_profits[-3] + 40),
    #     arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
    # )
    ax.set_ylim(0, 1000)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def run_variance_comparison_experiment(
    sim: Simulation,
    episodes=500,
    noise_levels=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
):
    """Compares different sub-assembly variance profiles sharing the SAME pack health."""
    print(
        f"Starting Sub-Assembly Variance Impact Experiment for {sim.base_product.name}..."
    )

    modes = ["uniform", "moderate", "extreme"]

    results_map = {mode: {"avg": [], "std": []} for mode in modes}

    for mode in modes:
        print(f"  Evaluating Profile: {mode}")
        for noise in noise_levels:
            metrics = sim.run_experiment(
                n_episodes=episodes, noise_level=noise, state_mode=mode
            )
            results_map[mode]["avg"].append(metrics["average_profit"])
            results_map[mode]["std"].append(metrics["std_profit"])

    # Plotting the comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["b", "g", "r"]
    markers = ["o", "s", "^"]

    for (name, metrics), c, m in zip(results_map.items(), colors, markers):
        ax.plot(
            noise_levels, metrics["avg"], marker=m, color=c, linewidth=2, label=name
        )
        # ax.fill_between(
        #     noise_levels,
        #     np.array(metrics["avg"]) - np.array(metrics["std"]),
        #     np.array(metrics["avg"]) + np.array(metrics["std"]),
        #     color=c,
        #     alpha=0.15,
        #     label=f"{name} ± STD",
        # )

    ax.set_title(
        f"Impact of Sub-Assembly Variance on Realized Profit for {sim.base_product.name}\n(All profiles have identical Pack Health = 0.80)",
        fontsize=14,
    )
    ax.set_xlabel("Sensor Noise (Standard Deviation)", fontsize=12)
    ax.set_ylabel("Average Realized Profit (£)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title="Module Health Distribution")

    # ax.annotate(
    #     "High variance unlocks highly profitable\nReuse options for specific modules.",
    #     xy=(0.0, results_map["Extreme [1.0, 1.0, 0.4]"][0]),
    #     xytext=(0.02, results_map["Extreme [1.0, 1.0, 0.4]"][0] - 30),
    #     arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
    # )
    ax.set_ylim(0, 1000)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def get_maximal_pathway(
    policy: dict,
    sadg: StateAugmentedDisassemblyGraph,
    state: tuple,
    params=None,
) -> str | list:
    """
    Recursively traverses the policy using the graph edges, handling both
    linear actions and disassembly branching.
    """
    if params is None:
        params = {
            "optimal_policy": [],
            "cost": [0.0],  # Start with a single float in a list to allow mutation
        }
    product, history = state

    # 1. Format the native tuple into the sorted policy string key
    if not history:
        policy_key = product
    else:
        # Sort the frozenset alphabetically to guarantee string consistency
        sorted_history = sorted(list(history))
        policy_key = f"{product}_[{','.join(sorted_history)}]"
    # Note: If your policy solver stringified the keys, you must either convert
    # current_key to that string format here, or update your solver to retain the raw tuples.
    if policy_key not in policy:
        return "No policy defined for this state"

    decision = policy[policy_key]

    if decision.startswith("CE:"):
        params["optimal_policy"].append(
            decision
        )  # Final CE cost is now handled in calculate_mdp_reward

    else:

        # 2. Parse the core action name (stripping out "Action: " if present)
        action_name = decision.split(": ")[-1] if ": " in decision else decision

        # 3. Use the graph to find the exact next state node(s)
        node = sadg.state_map.get(state)

        # Match the chosen action to the graph's generated edges to find the children
        next_nodes = [
            child
            for edge_action, child in node.children
            if edge_action.name == action_name
        ]

        action_obj = next(
            (a for a in node.product.actions if a.name == action_name), None
        )
        # 4. Handle Linear vs Branching Paths
        if not action_obj or not action_obj.is_disassembly:
            # Linear progression (e.g., Diagnostics, Wiping Data)

            next_state = (product, node.history.union({action_name}))
            params["cost"][0] += [
                a for a in node.product.actions if a.name == action_name
            ][0].cost
            params["optimal_policy"].append(action_name)
            get_maximal_pathway(
                policy=policy,
                sadg=sadg,
                state=next_state,
                params=params,
            )

        else:
            # Disassembly creates multiple branches. Recursively map all child components.
            # branched = True
            params["cost"][0] += [
                a for a in node.product.actions if a.name == action_name
            ][0].cost
            params["optimal_policy"].append(action_name)
            components = []
            for child in next_nodes[0]:
                child_state = (
                    child.product.name,
                    frozenset(node.history.union({action_name})),
                )
                child_params = {
                    "optimal_policy": [],
                    "cost": [0.0],
                }
                components.append(
                    [
                        child.product.name,
                        get_maximal_pathway(
                            policy=policy,
                            sadg=sadg,
                            state=child_state,
                            params=child_params,
                        ),
                    ]
                )
            # Store the structured component results instead of flattening
            params["components"] = components

    return params


def calculate_mdp_reward(
    policy,
    sadg,
    obs,
    true_state,
    alpha,
) -> list[dict]:
    """
    Calculates the economic and ecological scores for an optimal policy path.

    If the policy involves disassembly, this function returns a list of records,
    one for each final component. Otherwise, it returns a list with a single
    record for the top-level product.
    """
    if isinstance(policy, list):
        policy = dict(policy)

    opt_pol_params = get_maximal_pathway(policy=policy, sadg=sadg, state=sadg.root_key)

    records = []

    def find_and_process_ce_paths(
        params, base_cost, base_readable_path, base_actions_history, product_full_name
    ):
        """Recursively traverses policy params to find all CE leaf nodes."""

        if "_" in product_full_name:
            product_base_name = product_full_name.split("_")[-1]
        else:
            product_base_name = "laptop"

        # Add current node's actions to the path
        current_readable_path = [
            [product_base_name, action] for action in params["optimal_policy"]
        ]

        if params.get("components"):
            # This is a disassembly node. Recurse on children.
            new_base_cost = base_cost + params["cost"][0]
            new_readable_path = base_readable_path + current_readable_path
            new_history = base_actions_history.union(params["optimal_policy"])

            for comp_full_name, comp_params in params["components"]:
                find_and_process_ce_paths(
                    comp_params,
                    new_base_cost,
                    new_readable_path,
                    new_history,
                    comp_full_name,
                )
        else:
            # This is a terminal node with a CE decision. Create a record.
            total_cost = base_cost + params["cost"][0]
            marginal_cost = params["cost"][0]

            # Reconstruct the state to find the correct node for the CE decision
            actions_before_ce = params["optimal_policy"][:-1]
            history = base_actions_history.union(actions_before_ce)
            state = (product_full_name, history)
            node = sadg.state_map.get(state)

            if not node:
                return  # Should not happen with a valid policy

            true_h = true_state.get(product_full_name, obs.get(product_full_name, 0.5))
            observed_h = obs.get(product_full_name, 0.5)
            final_params = params.copy()
            final_params["cost"] = [total_cost]

            # observed health of the laptop OR components parts depending on policy action requirements
            ce_opt = next(
                opt
                for opt in node.ce_options
                if opt.name == final_params["optimal_policy"][-1].split(" ")[-1]
            )
            ce_route = ce_opt.name

            # Use the lower of true or observed health for reward calculation
            # this emulates asymetry of over classification vs under classification of health
            health = true_h if true_h < observed_h else observed_h

            mu = ce_opt.get_fuzzy_membership(health)  #   for fuzzy membership value

            ###T_econ
            # The final processing cost (e.g., logistics, admin) should be part of the CE option itself.
            # Here we assume a fixed cost for all options, but this could be customized per CE route.
            # Higher health = lower processing cost variability
            final_processing_cost = (50.0 * node.product.value_weight) + (
                np.random.randn()
                * ((50.0 * node.product.value_weight) * 0.2)
                * (1.0 - true_h)
            )

            # we should consider the MARGINAL profit of a component, assuming the shared disassembly
            # costs were justified by the sum of all component values.
            # The solver makes its decision based on this collective reward, so our reporting should reflect that.
            cost_for_reward_calc = marginal_cost + final_processing_cost

            # A better normalization is the component's own virgin value, not the whole laptop's.
            # This is approximated by its value weight multiplied by the total product value (alpha).
            # We also ensure it's not zero to avoid division errors.
            virgin_value = max(node.product.value_weight * alpha, 0.01)
            # virgin_value = node.product.value

            # Calculate expected resale value after value addition/recovery
            profitability = ce_opt.phi_k * ce_opt.base_value
            profit = profitability - cost_for_reward_calc
            t_econ = mu * np.sqrt(np.clip(profit / virgin_value, 0, 1))

            if t_econ > 1 or t_econ < 0:
                print(
                    f"Warning: T_Econ,k value {t_econ} is out of bounds. Clipping to [0, 1]."
                )
                t_econ = np.clip(t_econ, 0, 1)

            ###T_eco
            # virgin emissions for a laptop (kg CO2e)
            E_v = (
                331.0 + np.random.randn() * (331.0 * 0.2)
            ) * node.product.value_weight
            # A more standard formulation is to calculate the benefit relative to virgin emissions.
            emission_benefit = max(0.0, (E_v - ce_opt.E_k) / E_v)
            # Core equation
            t_eco = mu * ce_opt.Phi * emission_benefit

            if t_eco > 1 or t_eco < 0:
                print(
                    f"Warning: T_Eco,k value {t_eco} is out of bounds. Clipping to [0, 1]."
                )
                t_eco = np.clip(t_eco, 0, 1)

            mdp_reward = np.sqrt(t_econ**2 + t_eco**2)

            full_readable_path = base_readable_path + current_readable_path

            records.append(
                {
                    "component_name": product_base_name,
                    "readable_policy": json.dumps(full_readable_path),
                    "econ_reward": t_econ,
                    "eco_reward": t_eco,
                    "mdp_reward": mdp_reward,
                    "ce_route": ce_route,
                    "membership": mu,  # Fuzzy membership value
                    "cost": total_cost,  # This is the cost of the path to this CE option
                    "profitability": profitability,  # Potential resale value (phi * beta)
                    "profit": mu * profit,  # Actual profit in pounds
                    "base_value": ce_opt.base_value,  # The 'beta' value
                    "value_weight": node.product.value_weight,
                    "true_health": true_h,
                    "observed_health": observed_h,
                }
            )

    find_and_process_ce_paths(opt_pol_params, 0.0, [], frozenset(), sadg.root_key[0])
    return records


def run_mass_laptop_triage_experiment(sim, n_episodes=10000, noise_level=0.2):
    """
    Runs a Monte Carlo simulation for laptop ITAD triage.
    Generates instances, applies observation noise, varies market value, and solves the MDP.
    """
    print(
        f"Starting Mass Triage Experiment: {n_episodes} episodes, Noise Level: {noise_level}"
    )

    # Baseline market value defined in paper (£1000)
    BASE_ALPHA = 1000.0

    os.makedirs("results", exist_ok=True)
    filename = f"results/triage_experiment_n{n_episodes}_noise{noise_level}.csv"
    # If the file exists from a previous run, remove it to start fresh.
    if os.path.exists(filename):
        os.remove(filename)

    batch_results = []
    batch_size = 500  # Save to disk every 500 episodes

    for i in range(n_episodes):
        laptop_name = f"{i}"
        alpha = BASE_ALPHA * np.random.uniform(0.8, 1.2)
        laptop = Laptop(value=alpha, name=laptop_name)
        true_state = laptop.generate_true_state(mode="random")

        obs = {}
        for comp_name, true_health in true_state.items():
            noisy_health = np.random.normal(loc=true_health, scale=noise_level)
            obs[comp_name] = float(np.clip(noisy_health, 0.01, 1.0))

        sadg = StateAugmentedDisassemblyGraph(laptop)

        policy = sim.solver.solve(sadg.state_map[sadg.root_key], obs)

        # This now returns a list of dicts, one for each final component/product
        component_records = calculate_mdp_reward(policy, sadg, obs, true_state, alpha)

        for component_record in component_records:
            # Base record with info for the whole episode
            record = {
                "episode_id": i,
                "laptop_name": laptop_name,
                "alpha_value": round(alpha, 2),
                "true_state": json.dumps(
                    true_state
                ),  # Serialize dict to string for CSV
                "observation": json.dumps(obs),
                "full_policy": json.dumps(policy),
            }
            # Add the component-specific data
            record.update(component_record)
            batch_results.append(record)

        # Periodically write the batch to the CSV to manage memory
        if (i + 1) % batch_size == 0 or (i + 1) == n_episodes:
            if not batch_results:
                continue

            df_batch = pd.DataFrame(batch_results)
            # Write header only for the first chunk
            header = not os.path.exists(filename)
            df_batch.to_csv(filename, mode="a", header=header, index=False)

            batch_results = []  # Reset the batch
            print(f"Processed and saved episodes up to {i + 1}/{n_episodes}...")

    print(f"\nExperiment complete. Results saved to {filename}")
    # To maintain the function signature, we can read the file back.
    # Note: This reloads the entire dataset into memory. If memory is a
    # critical concern for the caller, consider returning the filename instead.
    # df = pd.read_csv(filename)
    # return df


if __name__ == "__main__":

    from battery import Battery
    from laptop import Laptop

    # Product = Battery(name="Test_Battery_Pack")
    # Product = Laptop(name="Test_Laptop")

    for i in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        Product = Laptop(name="Test_Laptop")
        sim = Simulation(product=Product)
        run_mass_laptop_triage_experiment(sim, n_episodes=1000, noise_level=i)

    # sim = Simulation(product=Product)

    # sim.inspect_dag()

    # sim.inspect_graph()

    # sim.run_single_trace(noise_level=0.1)

    # sim.plot_fuzzy_reward_curves()

    # run_noise_sensitivity_experiment(sim)

    # run_variance_comparison_experiment(sim)

    # 7. Final Blocking call
    print("All tests complete. Please close the plot windows to exit.")
    plt.show()
