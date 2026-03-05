from sim import Simulation
import matplotlib.pyplot as plt
import numpy as np


def run_noise_sensitivity_experiment():
    """Runs a sweep over different sensor noise levels and plots the result."""
    print("Starting Noise Sensitivity Monte Carlo Experiment...")
    sim = Simulation()

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    episodes_per_level = 500

    avg_profits = []
    std_profits = []

    for noise in noise_levels:
        print(f"  Running {episodes_per_level} episodes at noise std {noise:.2f}...")
        metrics = sim.run_experiment(n_episodes=episodes_per_level, noise_level=noise)
        avg_profits.append(metrics["average_profit"])
        std_profits.append(metrics["std_profit"])
        # print(f"Avg Profit for noise {noise}: £{metrics['average_profit']:.2f}")
        # print(f"Std Profit for noise {noise}: {metrics['std_profit']:.2f}")

    avg_profits = np.array(avg_profits)
    std_profits = np.array(std_profits)

    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 6))
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
        "Impact of Diagnostic Sensor Noise on Triage Profitability", fontsize=14
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

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def run_variance_comparison_experiment():
    """Compares different sub-assembly variance profiles sharing the SAME pack health."""
    print("Starting Sub-Assembly Variance Impact Experiment...")
    sim = Simulation()
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]  # , 0.50, 0.75, 1.0]
    episodes = 500

    # All 3 modes average out to exactly Pack Health = 0.80
    modes = {
        "Uniform [0.8, 0.8, 0.8]": "uniform",
        "Moderate [0.9, 0.8, 0.7]": "moderate",
        "Extreme [1.0, 1.0, 0.4]": "extreme",
    }

    results_map = {name: [] for name in modes}

    for name, mode in modes.items():
        print(f"  Evaluating Profile: {name}")
        for noise in noise_levels:
            metrics = sim.run_experiment(
                n_episodes=episodes, noise_level=noise, state_mode=mode
            )
            results_map[name].append(metrics["average_profit"])

    # Plotting the comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["b", "g", "r"]
    markers = ["o", "s", "^"]

    for (name, profits), c, m in zip(results_map.items(), colors, markers):
        ax.plot(noise_levels, profits, marker=m, color=c, linewidth=2, label=name)

    ax.set_title(
        "Impact of Sub-Assembly Variance on Realized Profit\n(All profiles have identical Pack Health = 0.80)",
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

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


if __name__ == "__main__":
    # 1. Initialize Simulator
    sim = Simulation()

    # 2. Inspect the Base Physical DAG
    sim.inspect_dag()

    # 3. Inspect the generated State-Augmented Graph (SAG)
    sim.inspect_graph()

    # 4. Print a detailed trace of a single triage decision
    # sim.run_single_trace(noise_level=0.1)

    sim.plot_fuzzy_reward_curves()

    # 5. Run the Original Monte Carlo batch (Modified to use "uniform" mode)
    # Note: Make sure your run_noise_sensitivity_experiment() calls
    # sim.run_experiment(..., state_mode="uniform") inside its loop!
    run_noise_sensitivity_experiment()

    # 6. Run the new Variance Comparison Experiment
    run_variance_comparison_experiment()

    # 7. Final Blocking call
    print("All tests complete. Please close the plot windows to exit.")
    plt.show()
