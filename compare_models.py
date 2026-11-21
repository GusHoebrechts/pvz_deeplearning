import argparse
import csv
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform

import gymnasium as gym
import gym_pvz  # Registers pvz-env-v2.
import numpy as np
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from a2c_masking import MaskedA2C


ROOT_DIR = Path(__file__).resolve().parent
MODEL_CLASSES = {
    "MaskablePPO": MaskablePPO,
    "MaskedA2C": MaskedA2C,
}
CHECKPOINTS = {
    "MaskablePPO": {
        "directory": Path("saved_models"),
        "best": "best_model.zip",
        "final": "maskable_ppo_pvz_final.zip",
    },
    "MaskedA2C": {
        "directory": Path("saved_models") / "a2c",
        "best": "best_model.zip",
        "final": "a2c_pvz_final.zip",
    },
}


def resolve_checkpoint(model_name, choice, root_dir=ROOT_DIR):
    if model_name not in CHECKPOINTS:
        raise ValueError(f"Unknown model: {model_name}")
    if choice not in ("best", "final"):
        raise ValueError(f"Unsupported checkpoint choice: {choice}")

    checkpoint = CHECKPOINTS[model_name]
    path = Path(root_dir) / checkpoint["directory"] / checkpoint[choice]
    if not path.is_file():
        raise FileNotFoundError(
            f"{model_name} {choice} checkpoint not found at {path}. "
            f"Train that model first or choose another checkpoint."
        )
    return path


def bootstrap_mean_ci(values, *, seed, n_resamples=10_000, confidence=0.95):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap confidence intervals need a non-empty vector.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    batch_size = 512
    for start in range(0, n_resamples, batch_size):
        end = min(start + batch_size, n_resamples)
        samples = rng.choice(
            values, size=(end - start, values.size), replace=True
        )
        means[start:end] = samples.mean(axis=1)

    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(low), float(high)


def validate_model_spaces(
    model_name, model, observation_space, action_space
):
    model_observation_space = model.observation_space
    if not isinstance(model_observation_space, spaces.Box) or not isinstance(
        observation_space, spaces.Box
    ):
        raise ValueError(
            f"{model_name} and pvz-env-v2 must both use Box observations."
        )
    if (
        model_observation_space.shape != observation_space.shape
        or model_observation_space.dtype != observation_space.dtype
        or not np.array_equal(
            model_observation_space.low, observation_space.low
        )
        or not np.array_equal(
            model_observation_space.high, observation_space.high
        )
    ):
        raise ValueError(
            f"{model_name} checkpoint expects observations with shape "
            f"{model_observation_space.shape} and dtype "
            f"{model_observation_space.dtype}; pvz-env-v2 provides shape "
            f"{observation_space.shape} and dtype {observation_space.dtype}. "
            "Retrain the checkpoint against the current environment before "
            "comparing it."
        )

    model_action_space = model.action_space
    if not isinstance(model_action_space, spaces.Discrete) or not isinstance(
        action_space, spaces.Discrete
    ):
        raise ValueError(
            f"{model_name} and pvz-env-v2 must both use Discrete actions."
        )
    if (
        model_action_space.n != action_space.n
        or model_action_space.start != action_space.start
    ):
        raise ValueError(
            f"{model_name} checkpoint has action space "
            f"{model_action_space}; pvz-env-v2 has {action_space}. "
            "Retrain the checkpoint against the current environment before "
            "comparing it."
        )


def evaluate_model(model_name, model, seeds):
    env = gym.make("pvz-env-v2")
    episode_results = []
    try:
        validate_model_spaces(
            model_name, model, env.observation_space, env.action_space
        )
        for episode_index, seed in enumerate(seeds, start=1):
            observation, _ = env.reset(seed=seed)
            episode_return = 0.0
            decision_steps = 0
            invalid_actions = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action_mask = np.asarray(
                    env.unwrapped.mask_available_actions(), dtype=bool
                )
                if action_mask.shape != (env.action_space.n,):
                    raise ValueError(
                        f"{model_name} received an action mask with shape "
                        f"{action_mask.shape}; expected {(env.action_space.n,)}."
                    )

                action, _ = model.predict(
                    observation,
                    action_masks=action_mask,
                    deterministic=True,
                )
                action = int(action)
                if not 0 <= action < action_mask.size:
                    raise ValueError(
                        f"{model_name} predicted out-of-range action {action}."
                    )
                invalid_actions += int(not action_mask[action])

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)
                episode_return += float(reward)
                decision_steps += 1

            episode_results.append(
                {
                    "model": model_name,
                    "episode": episode_index,
                    "seed": seed,
                    "episode_return": episode_return,
                    "survived": bool(env.unwrapped._scene.lives > 0),
                    "decision_steps": decision_steps,
                    "invalid_actions": invalid_actions,
                    "invalid_action_rate": (
                        invalid_actions / decision_steps if decision_steps else 0.0
                    ),
                }
            )
    finally:
        env.close()

    return episode_results


def summarize_model(model_name, checkpoint_path, episode_results, *, seed):
    returns = np.asarray(
        [result["episode_return"] for result in episode_results], dtype=float
    )
    survival = np.asarray(
        [result["survived"] for result in episode_results], dtype=float
    )
    decision_steps = np.asarray(
        [result["decision_steps"] for result in episode_results], dtype=float
    )
    total_steps = int(decision_steps.sum())
    invalid_actions = int(
        sum(result["invalid_actions"] for result in episode_results)
    )
    return_ci = bootstrap_mean_ci(returns, seed=seed)
    survival_ci = bootstrap_mean_ci(survival, seed=seed + 1)

    return {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "episodes": len(episode_results),
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "std_return": float(returns.std(ddof=1)) if returns.size > 1 else 0.0,
        "return_ci_95_low": return_ci[0],
        "return_ci_95_high": return_ci[1],
        "survival_rate": float(survival.mean()),
        "survival_ci_95_low": survival_ci[0],
        "survival_ci_95_high": survival_ci[1],
        "mean_decision_steps": float(decision_steps.mean()),
        "invalid_actions": invalid_actions,
        "invalid_action_rate": (
            invalid_actions / total_steps if total_steps else 0.0
        ),
    }


def compare_paired_results(
    ppo_results, a2c_results, *, seed, n_resamples=10_000
):
    ppo_by_seed = {result["seed"]: result for result in ppo_results}
    a2c_by_seed = {result["seed"]: result for result in a2c_results}
    if ppo_by_seed.keys() != a2c_by_seed.keys():
        raise ValueError("Paired comparisons require identical evaluation seeds.")

    paired_seeds = sorted(ppo_by_seed)
    reward_differences = np.asarray(
        [
            a2c_by_seed[item]["episode_return"]
            - ppo_by_seed[item]["episode_return"]
            for item in paired_seeds
        ],
        dtype=float,
    )
    survival_differences = np.asarray(
        [
            int(a2c_by_seed[item]["survived"])
            - int(ppo_by_seed[item]["survived"])
            for item in paired_seeds
        ],
        dtype=float,
    )
    reward_ci = bootstrap_mean_ci(
        reward_differences, seed=seed, n_resamples=n_resamples
    )
    survival_ci = bootstrap_mean_ci(
        survival_differences, seed=seed + 1, n_resamples=n_resamples
    )

    return {
        "paired_episodes": len(paired_seeds),
        "reward_difference_a2c_minus_ppo": {
            "mean": float(reward_differences.mean()),
            "ci_95_low": reward_ci[0],
            "ci_95_high": reward_ci[1],
        },
        "survival_difference_a2c_minus_ppo": {
            "mean": float(survival_differences.mean()),
            "ci_95_low": survival_ci[0],
            "ci_95_high": survival_ci[1],
        },
        "a2c_reward_wins": int(np.count_nonzero(reward_differences > 0)),
        "ppo_reward_wins": int(np.count_nonzero(reward_differences < 0)),
        "reward_ties": int(np.count_nonzero(reward_differences == 0)),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir, metadata, summaries, paired_comparison):
    report_lines = [
        "# Model evaluation comparison",
        "",
        f"- Episodes per model: {metadata['episodes']}",
        f"- Evaluation seeds: {metadata['seed_start']}–{metadata['seed_end']}",
        f"- Policy actions: deterministic",
        f"- Action masking: enabled for both models",
        f"- Device: {metadata['device']}",
        f"- Bootstrap resamples: {metadata['bootstrap_samples']:,}",
        "",
        "## Checkpoints",
        "",
    ]
    for model_name, checkpoint in metadata["checkpoints"].items():
        report_lines.append(f"- **{model_name}:** `{checkpoint}`")

    report_lines.extend(
        [
            "",
            "## Per-model results",
            "",
            "| Model | Mean return (95% bootstrap CI) | Median return | "
            "Survival rate (95% bootstrap CI) | Mean decisions | "
            "Invalid actions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        report_lines.append(
            f"| {summary['model']} "
            f"| {summary['mean_return']:.2f} "
            f"({summary['return_ci_95_low']:.2f}, "
            f"{summary['return_ci_95_high']:.2f}) "
            f"| {summary['median_return']:.2f} "
            f"| {summary['survival_rate']:.1%} "
            f"({summary['survival_ci_95_low']:.1%}, "
            f"{summary['survival_ci_95_high']:.1%}) "
            f"| {summary['mean_decision_steps']:.1f} "
            f"| {summary['invalid_actions']} |"
        )

    reward_difference = paired_comparison["reward_difference_a2c_minus_ppo"]
    survival_difference = paired_comparison[
        "survival_difference_a2c_minus_ppo"
    ]
    report_lines.extend(
        [
            "",
            "## Paired comparison (MaskedA2C − MaskablePPO)",
            "",
            f"- Mean reward difference: {reward_difference['mean']:.2f} "
            f"(95% paired bootstrap CI "
            f"{reward_difference['ci_95_low']:.2f} to "
            f"{reward_difference['ci_95_high']:.2f})",
            f"- Survival-rate difference: "
            f"{survival_difference['mean']:.1%} "
            f"(95% paired bootstrap CI "
            f"{survival_difference['ci_95_low']:.1%} to "
            f"{survival_difference['ci_95_high']:.1%})",
            f"- Per-seed reward outcomes: "
            f"A2C wins {paired_comparison['a2c_reward_wins']}, "
            f"PPO wins {paired_comparison['ppo_reward_wins']}, "
            f"ties {paired_comparison['reward_ties']}",
            "",
            "The confidence intervals describe evaluation variability; they are "
            "not a substitute for testing across independently trained model "
            "seeds. The per-episode CSV preserves the paired results for further "
            "analysis. Survival means the agent still has lives remaining when "
            "the episode ends.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the masked PPO and masked A2C checkpoints on paired "
            "PvZ evaluation seeds."
        )
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--ppo-checkpoint", choices=("best", "final"), default="best"
    )
    parser.add_argument(
        "--a2c-checkpoint", choices=("best", "final"), default="best"
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "auto"), default="cpu"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT_DIR
            / "comparison_results"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        ),
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")
    if args.seed_start < 0:
        parser.error("--seed-start cannot be negative.")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive.")
    return args


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir to preserve previous results."
        )

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    checkpoint_paths = {
        "MaskablePPO": resolve_checkpoint(
            "MaskablePPO", args.ppo_checkpoint
        ),
        "MaskedA2C": resolve_checkpoint("MaskedA2C", args.a2c_checkpoint),
    }

    models = {}
    for model_index, model_name in enumerate(MODEL_CLASSES):
        checkpoint_path = checkpoint_paths[model_name]
        print(f"Evaluating {model_name}: {checkpoint_path}")
        models[model_name] = MODEL_CLASSES[model_name].load(
            str(checkpoint_path), device=args.device
        )

    validation_env = gym.make("pvz-env-v2")
    try:
        for model_name, model in models.items():
            validate_model_spaces(
                model_name,
                model,
                validation_env.observation_space,
                validation_env.action_space,
            )
    finally:
        validation_env.close()

    all_results = []
    for model_name, model in models.items():
        results = evaluate_model(model_name, model, seeds)
        all_results.extend(results)

    summaries = []
    for model_index, model_name in enumerate(MODEL_CLASSES):
        model_results = [
            result for result in all_results if result["model"] == model_name
        ]
        summaries.append(
            summarize_model(
                model_name,
                checkpoint_paths[model_name],
                model_results,
                seed=args.seed_start + 100_000 + model_index * 10,
            )
        )

    ppo_results = [
        result for result in all_results if result["model"] == "MaskablePPO"
    ]
    a2c_results = [
        result for result in all_results if result["model"] == "MaskedA2C"
    ]
    paired_comparison = compare_paired_results(
        ppo_results,
        a2c_results,
        seed=args.seed_start + 200_000,
        n_resamples=args.bootstrap_samples,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_episode.csv", all_results)
    write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "paired_comparison.json").write_text(
        json.dumps(paired_comparison, indent=2), encoding="utf-8"
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "seed_end": seeds[-1],
        "device": args.device,
        "bootstrap_samples": args.bootstrap_samples,
        "deterministic_actions": True,
        "action_masking": True,
        "environment_id": "pvz-env-v2",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "stable_baselines3_version": version("stable-baselines3"),
        "sb3_contrib_version": version("sb3-contrib"),
        "checkpoints": {
            name: str(path) for name, path in checkpoint_paths.items()
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(output_dir, metadata, summaries, paired_comparison)

    print("\nPer-model summary:")
    for summary in summaries:
        print(
            f"{summary['model']}: mean return {summary['mean_return']:.2f} "
            f"(95% CI {summary['return_ci_95_low']:.2f} to "
            f"{summary['return_ci_95_high']:.2f}), "
            f"survival {summary['survival_rate']:.1%}, "
            f"invalid actions {summary['invalid_actions']}"
        )
    print(
        "\nPaired mean return difference (MaskedA2C - MaskablePPO): "
        f"{paired_comparison['reward_difference_a2c_minus_ppo']['mean']:.2f}"
    )
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
