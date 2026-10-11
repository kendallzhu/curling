"""Iterative self-play training for a multi-turn score network.

The notebook next to this module is intentionally just configuration and
display code.  This module owns the vectorized rollout, dataset retention,
checkpointing, and training loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil

import numpy as np

import curling_nn
import dataset
import nn
import scoring
import state
from bot import (
    DEFAULT_THROW_PERTURBATIONS, ThrowPerturbations, ThrowsGridSearcher,
    perturb_throws, select_robust_throws,
)
import sequential_training as seq


@dataclass(frozen=True)
class SelfPlayConfig:
    num_states: int = 500
    num_iterations: int = 30
    grid_size: tuple[int, int, int] = (10, 10, 4)
    state_generator_version: str = "turn_based"
    start_turn: int = 7
    halflife: float = 5.0
    max_stones: int = 10
    num_stones_per_side: int = 5
    hidden_layer_size: int = 20
    batch_size: int = 100
    training_iterations: int = 30
    learning_rate: float = 0.5
    validation_fraction: float = 0.0
    robust_top_fraction: float = 0.05
    perturbations: ThrowPerturbations = DEFAULT_THROW_PERTURBATIONS

    def retention_probability(self) -> float:
        if self.halflife <= 0:
            raise ValueError("halflife must be positive")
        return 2.0 ** (-1.0 / self.halflife)


def _copy_initial_training_stats(
    initial_model_path: str | Path, weights_dir: Path
) -> Path | None:
    """Copy the initial sequential checkpoint's evaluation record into this run."""
    checkpoint_path = Path(initial_model_path).resolve()
    if checkpoint_path.parent.name != "checkpoints":
        return None
    source_path = (
        checkpoint_path.parent.parent
        / "stats"
        / f"{checkpoint_path.stem}_evaluation.json"
    )
    if not source_path.is_file():
        return None
    destination_dir = weights_dir / "stats" / "initial_sequential_training"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / source_path.name
    shutil.copy2(source_path, destination_path)
    return destination_path


_pad_states = curling_nn.pad_sheet_states


def _model_scores(model, normalizer, sheet_states, max_stones: int) -> np.ndarray:
    features = seq.raw_padded_sheet_state_features(sheet_states, max_stones)
    outputs = model.run(normalizer.normalize(features)[:, :, None])
    return model.expected_score(outputs)


def _choose_throw(
    sheet_states: state.SheetStates,
    team: int,
    searcher: ThrowsGridSearcher,
    exact_score,
    perturbed_score,
    perturbations: ThrowPerturbations,
    robust_top_fraction: float,
) -> state.Throws:
    candidates, tiled_states = searcher.get_throws_for_num_sims(
        team=team, sheet_states=sheet_states
    )
    candidate_states = seq.cached_physics.run_until_stopping(
        sheet_states=tiled_states, throws=candidates
    )
    exact_scores = np.asarray(exact_score(candidate_states)).reshape(-1)
    if robust_top_fraction is None:
        num_states = sheet_states.x.shape[0]
        winners = exact_scores.reshape(-1, num_states).argmax(axis=0)
        indices = winners * num_states + np.arange(num_states)
        return state.Throws(
            angle_deg=candidates.angle_deg[indices], speed=candidates.speed[indices],
            turn=candidates.turn[indices], y_val=candidates.y_val[indices],
            team=candidates.team[indices],
        )

    def score_from_base(base_states, throws):
        return perturbed_score(
            seq.cached_physics.run_until_stopping(
                sheet_states=base_states, throws=throws
            )
        )

    return select_robust_throws(
        sheet_states=sheet_states,
        candidate_throws=candidates,
        exact_scores=exact_scores,
        scoring_function=score_from_base,
        perturbations=perturbations,
        top_fraction=robust_top_fraction,
    )


def generate_self_play_batch(
    model,
    normalizer: dataset.Normalizer,
    *,
    config: SelfPlayConfig,
    rng: np.random.Generator,
) -> tuple[seq.SequentialDataset, state.SheetStates]:
    """Generate labeled states from a sampled turn through the penultimate turn."""
    if config.state_generator_version != "turn_based":
        raise ValueError("self-play currently requires state_generator_version='turn_based'")
    if not 0 <= config.start_turn < config.max_stones:
        raise ValueError(
            f"start_turn must be in [0, {config.max_stones - 1}], got {config.start_turn}"
        )
    generator = seq.data_generation.generate_random_sheet_state_for_turn
    current = generator(
        turn=config.start_turn, num_sims=config.num_states, rng=rng
    )
    training_states = [current]
    searcher = ThrowsGridSearcher(*config.grid_size)

    def model_utility(states):
        scores = _model_scores(model, normalizer, _pad_states(states, config.max_stones), config.max_stones)
        return scores if team == 0 else -scores

    for _ in range(config.start_turn + 1, config.max_stones):
        team = int(current.next_team_to_play()[0])
        model_throw = _choose_throw(
            current, team, searcher, model_utility, model_utility,
            config.perturbations, config.robust_top_fraction,
        )
        current = seq.cached_physics.run_until_stopping(
            sheet_states=current,
            throws=perturb_throws(model_throw, config.perturbations, rng),
        )
        training_states.append(current)

    team = int(current.next_team_to_play()[0])

    def actual_utility(states):
        return scoring.get_net_score_for_team(states, team)

    final_throw = _choose_throw(
        current, team, searcher, actual_utility, actual_utility,
        config.perturbations, config.robust_top_fraction,
    )
    final_state = seq.cached_physics.run_until_stopping(
        sheet_states=current,
        throws=perturb_throws(final_throw, config.perturbations, rng),
    )
    final_scores = scoring.get_net_score_for_team(final_state, 0)

    rows = state.concat(
        [_pad_states(states, config.max_stones) for states in training_states]
    )
    labels = np.tile(final_scores, len(training_states))
    return (
        seq.SequentialDataset(
            rows, labels, config.max_stones, config.num_stones_per_side
        ),
        final_state,
    )


def retain_and_combine(
    previous: seq.SequentialDataset,
    new_data: seq.SequentialDataset,
    *,
    halflife: float,
    rng: np.random.Generator,
) -> seq.SequentialDataset:
    """Return D_i plus independently retained rows from X_(i-1)."""
    probability = 2.0 ** (-1.0 / halflife)
    keep = rng.random(previous.final_scores.size) < probability
    retained = state.take_sheet_states(previous.sheet_states, np.flatnonzero(keep))
    combined_states = state.concat([new_data.sheet_states, retained])
    combined_scores = np.concatenate([new_data.final_scores, previous.final_scores[keep]])
    return seq.SequentialDataset(
        combined_states, combined_scores, new_data.max_stones, new_data.num_stones_per_side
    )


def train_from_weights(
    model,
    data: seq.SequentialDataset,
    *,
    config: SelfPlayConfig,
    seed: int,
) -> tuple[object, dataset.Normalizer, list[dict[str, float | int | None]]]:
    """Train in-place from the supplied checkpoint using the old schedule."""
    training_data = data.training_data()
    loss = nn.SoftmaxCrossEntropyLoss()
    targets = np.asarray(data.final_scores, dtype=float)
    epoch_metrics = []
    for iteration in range(config.training_iterations):
        learning_rate = config.learning_rate * (
            0.5 * (1.0 + np.cos(np.pi * iteration / config.training_iterations))
        )
        for batch in training_data.shuffle_batches(config.batch_size, seed=seed + iteration):
            model.train_batched(batch, loss, learning_rate, 0)
        cross_entropy = float(model.get_average_loss_batched(
            training_data.input_features, training_data.answers, loss
        ))
        predicted_scores = model.expected_score(
            model.run(training_data.input_features[:, :, None])
        ).reshape(-1)
        residuals = targets - predicted_scores
        mse = float(np.mean(residuals**2))
        total_sum_squares = float(np.sum((targets - np.mean(targets)) ** 2))
        r_squared = (
            float(1.0 - np.sum(residuals**2) / total_sum_squares)
            if total_sum_squares > 0 else None
        )
        epoch_metrics.append({
            "epoch": iteration + 1,
            "cross_entropy": cross_entropy,
            "r_squared": r_squared,
            "mse": mse,
        })
    return model, training_data.normalizer, epoch_metrics


def evaluate_checkpoint_on_training_data(
    checkpoint_path: str | Path, dataset_path: str | Path
) -> dict[str, float | int | None]:
    """Evaluate a saved self-play checkpoint on its saved combined X_i data."""
    model, normalizer = seq.load_model(checkpoint_path)
    data = seq.load_sequential_dataset(dataset_path)
    training_data = data.training_data()
    input_features = normalizer.normalize(training_data.raw_inputs)
    targets = np.asarray(data.final_scores, dtype=float)
    answers = (
        targets[:, None]
        == np.arange(-data.num_stones_per_side, data.num_stones_per_side + 1)
    ).astype(float)
    cross_entropy = float(model.get_average_loss_batched(
        input_features, answers, nn.SoftmaxCrossEntropyLoss()
    ))
    predicted_scores = model.expected_score(
        model.run(input_features[:, :, None])
    ).reshape(-1)
    residuals = targets - predicted_scores
    mse = float(np.mean(residuals**2))
    total_sum_squares = float(np.sum((targets - np.mean(targets)) ** 2))
    r_squared = (
        float(1.0 - np.sum(residuals**2) / total_sum_squares)
        if total_sum_squares > 0 else None
    )
    return {
        "training_rows": int(targets.size),
        "cross_entropy": cross_entropy,
        "r_squared": r_squared,
        "mse": mse,
    }


def run_self_play(
    *,
    config: SelfPlayConfig,
    artifact_dir: str | Path,
    initial_model_path: str | Path,
    initial_dataset_path: str | Path,
    seed: int = 0,
    resume_from: int | None = None,
    weights_dir: str | Path | None = None,
    notebook_parameters: dict | None = None,
) -> dict[int, dict[str, object]]:
    """Run self-play, retaining datasets in artifact_dir and weights in weights_dir."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = (
        seq.ensure_model_checkpoint_run_dir(weights_dir)
        if weights_dir is not None else None
    )
    saved_iterations: dict[str, dict[str, object]] = {}
    saved_checkpoint_metrics: dict[str, dict[str, object]] = {}
    if weights_dir is not None:
        stats_dir = weights_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        initial_training_stats_path = _copy_initial_training_stats(
            initial_model_path, weights_dir
        )
        repo_root = Path(seq.__file__).resolve().parent

        def input_label(path: str | Path) -> str:
            resolved = Path(path).resolve()
            try:
                return str(resolved.relative_to(repo_root))
            except ValueError:
                return f"external:{resolved.name}"

        seq.save_metadata(weights_dir / "run_config.json", {
            "run_config_version": 2,
            "category": weights_dir.parent.name,
            "run_name": weights_dir.name,
            "seed": seed,
            "config": asdict(config),
            "notebook_parameters": notebook_parameters,
            "training_data_generation": {
                "state_generator_version": config.state_generator_version,
                "sampled_start_turn": config.start_turn,
                "sampled_start_states_per_iteration": config.num_states,
                "model_controlled_turns": list(
                    range(config.start_turn + 1, config.max_stones)
                ),
                "final_turn_policy": "actual-score throw selection",
                "labeled_states_per_start_state": (
                    config.max_stones - config.start_turn
                ),
                "new_rows_per_iteration": (
                    (config.max_stones - config.start_turn) * config.num_states
                ),
                "label": (
                    "Final net score for team 0, copied to states at turns "
                    f"{config.start_turn} through {config.max_stones - 1}."
                ),
                "retained_previous_row_probability": config.retention_probability(),
                "throw_search": {
                    "grid_size": list(config.grid_size),
                    "robust_top_fraction": config.robust_top_fraction,
                    "perturbations": asdict(config.perturbations),
                },
            },
            "network_training": {
                "hidden_layer_size": config.hidden_layer_size,
                "batch_size": config.batch_size,
                "iterations_per_checkpoint": config.training_iterations,
                "loss": "softmax cross-entropy",
                "optimizer": "mini-batch gradient descent",
                "regularization": 0.0,
                "learning_rate": config.learning_rate,
                "learning_rate_schedule": (
                    "learning_rate * 0.5 * (1 + cos(pi * iteration / iterations_per_checkpoint))"
                ),
                "normalizer": "refit from the combined training data for each checkpoint",
                "seed": "run seed + checkpoint index; batch shuffle uses seed + epoch",
            },
            "initial_model": input_label(initial_model_path),
            "initial_dataset": input_label(initial_dataset_path),
            "initial_training_stats": (
                initial_training_stats_path.relative_to(weights_dir).as_posix()
                if initial_training_stats_path is not None else None
            ),
        })
        stats_path = stats_dir / "training_stats.json"
        legacy_eval_stats_dir = weights_dir / "eval_stats"
        legacy_root_stats_path = weights_dir / "training_stats.json"
        training_metrics_path = stats_dir / "training_metrics.json"
        if resume_from is not None:
            resume_stats_path = next(
                (path for path in (
                    stats_path,
                    legacy_eval_stats_dir / "training_stats.json",
                    legacy_root_stats_path,
                ) if path.exists()),
                None,
            )
            if resume_stats_path is not None:
                saved_iterations = json.loads(
                    resume_stats_path.read_text()
                ).get("iterations", {})
        legacy_metrics_path = legacy_eval_stats_dir / "training_metrics.json"
        if resume_from is not None and (
            training_metrics_path.exists() or legacy_metrics_path.exists()
        ):
            metrics_path = (
                training_metrics_path
                if training_metrics_path.exists()
                else legacy_metrics_path
            )
            saved_checkpoint_metrics = json.loads(
                metrics_path.read_text()
            ).get("checkpoints", {})

    if resume_from is None:
        model, normalizer = seq.load_model(initial_model_path)
        previous = seq.load_sequential_dataset(initial_dataset_path)
        start = 1
    else:
        checkpoint_path = artifact_dir / f"m_{resume_from}.npz"
        if weights_dir is not None:
            for candidate in (
                weights_dir / "checkpoints" / f"m_{resume_from}.npz",
                weights_dir / f"m_{resume_from}.npz",
                checkpoint_path,
            ):
                if candidate.exists():
                    checkpoint_path = candidate
                    break
        model, normalizer = seq.load_model(checkpoint_path)
        previous = seq.load_sequential_dataset(artifact_dir / f"X_{resume_from}.npz")
        start = resume_from + 1
    if previous.sheet_states.x.shape[1] != config.max_stones:
        previous = seq.SequentialDataset(
            _pad_states(previous.sheet_states, config.max_stones),
            previous.final_scores,
            config.max_stones,
            config.num_stones_per_side,
        )

    results = {}
    for i in range(start, config.num_iterations + 1):
        rng = np.random.default_rng(seed + i)
        generated, _ = generate_self_play_batch(model, normalizer, config=config, rng=rng)
        combined = retain_and_combine(previous, generated, halflife=config.halflife, rng=rng)
        model, normalizer, epoch_metrics = train_from_weights(
            model, combined, config=config, seed=seed + i
        )
        losses = [float(entry["cross_entropy"]) for entry in epoch_metrics]
        r_squared = [entry["r_squared"] for entry in epoch_metrics]
        mse = [float(entry["mse"]) for entry in epoch_metrics]
        seq.write_sequential_dataset(artifact_dir / f"D_{i}.npz", generated)
        seq.write_sequential_dataset(artifact_dir / f"X_{i}.npz", combined)
        checkpoint_dir = (
            weights_dir / "checkpoints" if weights_dir is not None else artifact_dir
        )
        seq.write_model(checkpoint_dir / f"m_{i}.npz", model, normalizer)
        record = {
            "new_rows": generated.final_scores.size,
            "training_rows": combined.final_scores.size,
            "loss": losses,
            "r_squared": r_squared,
            "mse": mse,
            "epoch_metrics": epoch_metrics,
        }
        results[i] = record
        if weights_dir is not None:
            saved_iterations[str(i)] = {
                **record,
                "final_training_loss": losses[-1] if losses else None,
            }
            seq.save_metadata(stats_path, {
                "category": weights_dir.parent.name,
                "run_name": weights_dir.name,
                "seed": seed,
                "config": asdict(config),
                "iterations": saved_iterations,
            })
            final_metrics = epoch_metrics[-1] if epoch_metrics else {}
            saved_checkpoint_metrics[str(i)] = {
                "training_rows": int(combined.final_scores.size),
                "cross_entropy": final_metrics.get("cross_entropy"),
                "r_squared": final_metrics.get("r_squared"),
                "mse": final_metrics.get("mse"),
                "epoch_metrics": epoch_metrics,
            }
            seq.save_metadata(training_metrics_path, {
                "category": weights_dir.parent.name,
                "run_name": weights_dir.name,
                "metric_scope": (
                    "Training-set fit on retained X_i data; R-squared is for "
                    "predicted expected score versus final score."
                ),
                "checkpoints": saved_checkpoint_metrics,
            })
        previous = combined
    return results


def _greedy_model_throw(sheet_states, model, normalizer, searcher, max_stones):
    team = int(sheet_states.next_team_to_play()[0])

    def utility(states):
        scores = _model_scores(
            model, normalizer, _pad_states(states, max_stones), max_stones
        )
        return scores if team == 0 else -scores

    return _choose_throw(
        sheet_states,
        team,
        searcher,
        utility,
        utility,
        ThrowPerturbations(angle_offsets_deg=(0.0,), probabilities=(1.0,)),
        None,
    )


def _greedy_grid_throw(sheet_states, searcher):
    team = int(sheet_states.next_team_to_play()[0])

    def utility(states):
        return scoring.get_net_score_for_team(states, team)

    return _choose_throw(
        sheet_states,
        team,
        searcher,
        utility,
        utility,
        ThrowPerturbations(angle_offsets_deg=(0.0,), probabilities=(1.0,)),
        None,
    )


def _evaluate_policy_group(
    initial_states: state.SheetStates,
    model,
    normalizer: dataset.Normalizer,
    *,
    model_turns: frozenset[int],
    model_team: int,
    grid_size: tuple[int, int, int],
    max_stones: int,
) -> np.ndarray:
    """Roll out one half of the paired evaluation from fixed initial states."""
    scores, _ = _rollout_policy_group(
        initial_states,
        model,
        normalizer,
        model_turns=model_turns,
        model_team=model_team,
        grid_size=grid_size,
        max_stones=max_stones,
        record_throws=False,
    )
    return scores


def _rollout_policy_group(
    initial_states: state.SheetStates,
    model,
    normalizer: dataset.Normalizer,
    *,
    model_turns: frozenset[int],
    model_team: int,
    grid_size: tuple[int, int, int],
    max_stones: int,
    record_throws: bool,
    perturbations: ThrowPerturbations = DEFAULT_THROW_PERTURBATIONS,
    perturbation_seed: int = 0,
) -> tuple[np.ndarray, list[list[state.Throw]] | None]:
    """Roll out a policy group and optionally retain each game's throws."""
    # Fixed seed gives every policy the same noise draw per game and turn.
    perturbation_rng = np.random.default_rng(perturbation_seed)
    current = initial_states
    searcher = ThrowsGridSearcher(*grid_size)
    throw_history = [[] for _ in range(initial_states.x.shape[0])] if record_throws else None
    # The supplied state contains all stones from earlier turns. This also
    # lets parity evaluations start before turn 6 while preserving the existing
    # turn-7-through-10 single-turn comparisons.
    first_turn = initial_states.x.shape[1] + 1
    for turn in range(first_turn, max_stones + 1):
        if turn in model_turns:
            throws = _greedy_model_throw(
                current, model, normalizer, searcher, max_stones
            )
        else:
            throws = _greedy_grid_throw(current, searcher)
        throws = perturb_throws(throws, perturbations, perturbation_rng)
        if throw_history is not None:
            for game_index in range(len(throw_history)):
                throw_history[game_index].append(throws.get_throw(game_index))
        current = seq.cached_physics.run_until_stopping(
            sheet_states=current, throws=throws
        )
    return scoring.get_net_score_for_team(current, model_team), throw_history


def evaluate_checkpoints_against_greedy_with_games(
    *,
    checkpoint_paths: dict[str, str | Path],
    num_states: int = 200,
    seed: int = 2026,
    grid_size: tuple[int, int, int] = (10, 10, 4),
    max_stones: int = 10,
    states: state.SheetStates | None = None,
    parity_states: state.SheetStates | None = None,
) -> list[dict[str, object]]:
    """Return paired game states, throws, and scores for interactive replay.

    Existing single-turn groups use states through turn 6, and the added
    parity groups use states through turn 5 so they can include turns 6 and 7.
    The same greedy policy controls turn 10 and all turns outside each
    ``model_turns`` set. Each result contains ``num_states // 2`` games.
    """
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty")
    if num_states < 2 or num_states % 2:
        raise ValueError("num_states must be a positive even number")
    if states is None:
        states = seq.data_generation.generate_random_sheet_state_for_turn(
            turn=6, num_sims=num_states, rng=np.random.default_rng(seed)
        )
    if states.x.shape[0] != num_states:
        raise ValueError("states must contain num_states rows")
    if parity_states is None:
        parity_states = seq.data_generation.generate_random_sheet_state_for_turn(
            turn=5, num_sims=num_states, rng=np.random.default_rng(seed + 1)
        )
    if parity_states.x.shape[0] != num_states:
        raise ValueError("parity_states must contain num_states rows")
    if parity_states.x.shape[1] != 5:
        raise ValueError("parity_states must represent boards through turn 5")

    midpoint = num_states // 2
    first_half = state.take_sheet_states(states, np.arange(midpoint))
    second_half = state.take_sheet_states(states, np.arange(midpoint, num_states))
    parity_midpoint = num_states // 2
    parity_first_half = state.take_sheet_states(
        parity_states, np.arange(parity_midpoint)
    )
    parity_second_half = state.take_sheet_states(
        parity_states, np.arange(parity_midpoint, num_states)
    )
    groups = (
        ("model_turn_7", frozenset((7,)), 0, first_half, "single_odd"),
        ("model_turn_9", frozenset((9,)), 0, first_half, "single_odd"),
        ("model_turn_8", frozenset((8,)), 1, second_half, "single_even"),
        (
            "model_turns_7_9",
            frozenset((7, 9)),
            0,
            parity_first_half,
            "parity_odd",
        ),
        (
            "model_turns_6_8",
            frozenset((6, 8)),
            1,
            parity_second_half,
            "parity_even",
        ),
    )

    greedy_results = {}
    for baseline_name, model_team, group_states in (
        ("single_odd", 0, first_half),
        ("single_even", 1, second_half),
        ("parity_odd", 0, parity_first_half),
        ("parity_even", 1, parity_second_half),
    ):
        scores, throws = _rollout_policy_group(
            group_states,
            model=None,
            normalizer=None,
            model_turns=frozenset(),
            model_team=model_team,
            grid_size=grid_size,
            max_stones=max_stones,
            record_throws=True,
        )
        greedy_results[baseline_name] = (scores, throws)

    records: list[dict[str, object]] = []
    for run_name, checkpoint_path in checkpoint_paths.items():
        model, normalizer = seq.load_model(checkpoint_path)
        for group_name, model_turns, model_team, group_states, baseline_name in groups:
            model_scores, model_throws = _rollout_policy_group(
                group_states,
                model,
                normalizer,
                model_turns=model_turns,
                model_team=model_team,
                grid_size=grid_size,
                max_stones=max_stones,
                record_throws=True,
            )
            greedy_scores, greedy_throws = greedy_results[baseline_name]
            records.append({
                "run": run_name,
                "group": group_name,
                "model_team": model_team,
                "model_turns": tuple(sorted(model_turns)),
                "games": [
                    {
                        "initial_state": state.take_sheet_states(
                            group_states, np.array([game_index])
                        ),
                        "model_throws": model_throws[game_index],
                        "greedy_throws": greedy_throws[game_index],
                        "model_score": float(model_scores[game_index]),
                        "greedy_score": float(greedy_scores[game_index]),
                    }
                    for game_index in range(len(model_scores))
                ],
            })
    return records


def summarize_evaluation_games(
    records: list[dict[str, object]],
) -> list[dict[str, float | int | str]]:
    """Summarize the per-game records returned by the paired replay evaluation."""
    summaries = []
    for record in records:
        games = record["games"]
        model_scores = np.asarray([game["model_score"] for game in games], dtype=float)
        greedy_scores = np.asarray([game["greedy_score"] for game in games], dtype=float)
        differences = model_scores - greedy_scores
        standard_error = float(np.std(differences, ddof=1) / np.sqrt(differences.size))
        average_advantage = float(np.mean(differences))
        summaries.append({
            "run": record["run"],
            "group": record["group"],
            "model_team": record["model_team"],
            "num_states": int(differences.size),
            "average_model_score": float(np.mean(model_scores)),
            "average_greedy_score": float(np.mean(greedy_scores)),
            "average_advantage": average_advantage,
            "advantage_standard_error": standard_error,
            "advantage_ci95_low": average_advantage - 1.96 * standard_error,
            "advantage_ci95_high": average_advantage + 1.96 * standard_error,
            "win_rate": float(np.mean(differences > 0)),
            "tie_rate": float(np.mean(differences == 0)),
            "loss_rate": float(np.mean(differences < 0)),
        })
    return summaries


def save_evaluation_games(
    path: str | Path,
    records: list[dict[str, object]],
    *,
    parameters: dict[str, object],
) -> None:
    """Save replay records as a compressed, pickle-free NPZ archive."""
    archive: dict[str, np.ndarray] = {}
    record_metadata = []
    for record_index, record in enumerate(records):
        games = record["games"]
        prefix = f"record_{record_index}_"
        record_metadata.append({
            "run": record["run"],
            "group": record["group"],
            "model_team": record["model_team"],
            "model_turns": list(record["model_turns"]),
            "prefix": prefix,
            "num_games": len(games),
        })

        initial_states = [game["initial_state"] for game in games]
        archive[prefix + "first_team"] = np.concatenate(
            [states.first_team for states in initial_states]
        )
        for key, get_array in (
            ("x", lambda states: states.x),
            ("y", lambda states: states.y),
            ("velocity_v", lambda states: states.velocities.v),
            ("velocity_theta", lambda states: states.velocities.theta),
            ("rotation_directions", lambda states: states.rotation_directions),
        ):
            archive[prefix + key] = np.concatenate(
                [get_array(states) for states in initial_states]
            )

        archive[prefix + "model_score"] = np.asarray(
            [game["model_score"] for game in games], dtype=float
        )
        archive[prefix + "greedy_score"] = np.asarray(
            [game["greedy_score"] for game in games], dtype=float
        )
        for policy_key in ("model_throws", "greedy_throws"):
            throws = [game[policy_key] for game in games]
            for field in ("angle_deg", "speed", "turn", "y_val", "team"):
                archive[prefix + policy_key + "_" + field] = np.asarray(
                    [[getattr(throw, field) for throw in game_throws] for game_throws in throws]
                )

    metadata = {
        "schema_version": 1,
        "parameters": parameters,
        "records": record_metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata)), **archive)


def load_evaluation_games(path: str | Path) -> list[dict[str, object]]:
    """Load replay records saved by :func:`save_evaluation_games`."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("schema_version") != 1:
            raise ValueError(f"Unsupported self-play evaluation archive: {path}")
        records = []
        for record_meta in metadata["records"]:
            prefix = record_meta["prefix"]
            count = record_meta["num_games"]
            arrays = {
                key: np.array(data[prefix + key], copy=True)
                for key in (
                    "first_team", "x", "y", "velocity_v", "velocity_theta",
                    "rotation_directions", "model_score", "greedy_score",
                )
            }

            throw_fields = {
                policy_key: {
                    name: np.array(data[prefix + policy_key + "_" + name], copy=True)
                    for name in ("angle_deg", "speed", "turn", "y_val", "team")
                }
                for policy_key in ("model_throws", "greedy_throws")
            }

            def load_throws(policy_key: str, game_index: int) -> list[state.Throw]:
                fields = throw_fields[policy_key]
                return [
                    state.Throw(
                        angle_deg=float(fields["angle_deg"][game_index, throw_index]),
                        speed=float(fields["speed"][game_index, throw_index]),
                        turn=int(fields["turn"][game_index, throw_index]),
                        y_val=float(fields["y_val"][game_index, throw_index]),
                        team=int(fields["team"][game_index, throw_index]),
                    )
                    for throw_index in range(fields["turn"].shape[1])
                ]

            games = []
            for game_index in range(count):
                games.append({
                    "initial_state": state.SheetStates(
                        first_team=arrays["first_team"][game_index : game_index + 1],
                        x=arrays["x"][game_index : game_index + 1],
                        y=arrays["y"][game_index : game_index + 1],
                        velocities=state.Velocities(
                            v=arrays["velocity_v"][game_index : game_index + 1],
                            theta=arrays["velocity_theta"][game_index : game_index + 1],
                        ),
                        rotation_directions=arrays["rotation_directions"][game_index : game_index + 1],
                    ),
                    "model_throws": load_throws("model_throws", game_index),
                    "greedy_throws": load_throws("greedy_throws", game_index),
                    "model_score": float(arrays["model_score"][game_index]),
                    "greedy_score": float(arrays["greedy_score"][game_index]),
                })
            records.append({
                "run": record_meta["run"],
                "group": record_meta["group"],
                "model_team": record_meta["model_team"],
                "model_turns": tuple(record_meta["model_turns"]),
                "games": games,
            })
    return records


def evaluate_checkpoints(
    *,
    checkpoint_dir: str | Path,
    checkpoint_indices: tuple[int, ...] = (1, 2, 5, 10, 20, 30),
    num_states: int = 1000,
    seed: int = 0,
    grid_size: tuple[int, int, int] = (10, 10, 4),
    state_generator_version: str = "turn_based",
    max_stones: int = 10,
    states: state.SheetStates | None = None,
) -> list[dict[str, float | int | str]]:
    """Evaluate checkpoints on the same 1,000 six-throw states.

    The first half uses the model on turns 7 and 9; the second half uses it on
    turn 8.  In both cases turns 10 and the remaining turns use actual-score
    greedy grid search.  Scores and standard errors are from the model team's
    perspective.
    """
    if state_generator_version != "turn_based":
        raise ValueError("evaluation requires state_generator_version='turn_based'")
    if num_states < 2 or num_states % 2:
        raise ValueError("num_states must be a positive even number")
    if states is None:
        states = seq.data_generation.generate_random_sheet_state_for_turn(
            turn=6, num_sims=num_states, rng=np.random.default_rng(seed)
        )
    if states.x.shape[0] != num_states:
        raise ValueError("states must contain num_states rows")

    midpoint = num_states // 2
    first_half = state.take_sheet_states(states, np.arange(midpoint))
    second_half = state.take_sheet_states(states, np.arange(midpoint, num_states))
    records = []
    for checkpoint in checkpoint_indices:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_path = checkpoint_dir / "checkpoints" / f"m_{checkpoint}.npz"
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / f"m_{checkpoint}.npz"
        model, normalizer = seq.load_model(
            checkpoint_path
        )
        scores_a = _evaluate_policy_group(
            first_half,
            model,
            normalizer,
            model_turns=frozenset((7, 9)),
            model_team=0,
            grid_size=grid_size,
            max_stones=max_stones,
        )
        scores_b = _evaluate_policy_group(
            second_half,
            model,
            normalizer,
            model_turns=frozenset((8,)),
            model_team=1,
            grid_size=grid_size,
            max_stones=max_stones,
        )
        for group, scores in (("model_turns_7_9", scores_a), ("model_turn_8", scores_b)):
            records.append({
                "checkpoint": checkpoint,
                "group": group,
                "model_team": 0 if group == "model_turns_7_9" else 1,
                "num_states": int(scores.size),
                "average_score": float(np.mean(scores)),
                "standard_error": float(np.std(scores, ddof=1) / np.sqrt(scores.size)),
            })
    return records


def evaluate_all_greedy(
    *,
    checkpoint_indices: tuple[int, ...] = (1, 2, 5, 10, 20, 30),
    num_states: int = 1000,
    seed: int = 0,
    grid_size: tuple[int, int, int] = (10, 10, 4),
    state_generator_version: str = "turn_based",
    states: state.SheetStates | None = None,
) -> list[dict[str, float | int | str]]:
    """Run the equivalent fixed-state baseline with greedy search everywhere."""
    if state_generator_version != "turn_based":
        raise ValueError("evaluation requires state_generator_version='turn_based'")
    if num_states < 2 or num_states % 2:
        raise ValueError("num_states must be a positive even number")
    if states is None:
        states = seq.data_generation.generate_random_sheet_state_for_turn(
            turn=6, num_sims=num_states, rng=np.random.default_rng(seed)
        )
    midpoint = num_states // 2
    groups = (
        ("model_turns_7_9", 0, state.take_sheet_states(states, np.arange(midpoint))),
        ("model_turn_8", 1, state.take_sheet_states(states, np.arange(midpoint, num_states))),
    )
    baseline = []
    for group, model_team, group_states in groups:
        scores = _evaluate_policy_group(
            group_states,
            model=None,
            normalizer=None,
            model_turns=frozenset(),
            model_team=model_team,
            grid_size=grid_size,
            max_stones=10,
        )
        baseline.append((group, model_team, float(np.mean(scores)), float(
            np.std(scores, ddof=1) / np.sqrt(scores.size)
        )))
    records = []
    for checkpoint in checkpoint_indices:
        for group, model_team, average_score, standard_error in baseline:
            records.append({
                "checkpoint": checkpoint,
                "group": group,
                "model_team": model_team,
                "num_states": num_states // 2,
                "average_score": average_score,
                "standard_error": standard_error,
            })
    return records


def evaluate_turn9_vs_turn8(
    *,
    checkpoint_paths: dict[int, str | Path],
    num_states: int = 1000,
    seed: int = 0,
    grid_size: tuple[int, int, int] = (10, 10, 4),
    state_generator_version: str = "turn_based",
    max_stones: int = 10,
    states: state.SheetStates | None = None,
) -> list[dict[str, float | int | str]]:
    """Compare a model used only on turn 9 with one used only on turn 8."""
    if state_generator_version != "turn_based":
        raise ValueError("evaluation requires state_generator_version='turn_based'")
    if num_states < 2 or num_states % 2:
        raise ValueError("num_states must be a positive even number")
    if states is None:
        states = seq.data_generation.generate_random_sheet_state_for_turn(
            turn=6, num_sims=num_states, rng=np.random.default_rng(seed)
        )
    midpoint = num_states // 2
    first_half = state.take_sheet_states(states, np.arange(midpoint))
    second_half = state.take_sheet_states(states, np.arange(midpoint, num_states))
    records = []
    for checkpoint, path in checkpoint_paths.items():
        model, normalizer = seq.load_model(path)
        scores_9 = _evaluate_policy_group(
            first_half, model, normalizer, model_turns=frozenset((9,)),
            model_team=0, grid_size=grid_size, max_stones=max_stones,
        )
        scores_8 = _evaluate_policy_group(
            second_half, model, normalizer, model_turns=frozenset((8,)),
            model_team=1, grid_size=grid_size, max_stones=max_stones,
        )
        for group, model_team, scores in (
            ("model_turn_9", 0, scores_9),
            ("model_turn_8", 1, scores_8),
        ):
            records.append({
                "checkpoint": checkpoint,
                "group": group,
                "model_team": model_team,
                "num_states": int(scores.size),
                "average_score": float(np.mean(scores)),
                "standard_error": float(np.std(scores, ddof=1) / np.sqrt(scores.size)),
            })
    return records


def evaluate_checkpoints_against_greedy(
    *,
    checkpoint_paths: dict[str, str | Path],
    num_states: int = 200,
    seed: int = 2026,
    grid_size: tuple[int, int, int] = (10, 10, 4),
    max_stones: int = 10,
    states: state.SheetStates | None = None,
) -> list[dict[str, float | int | str]]:
    """Compare checkpoints with all-greedy play on paired fixed states.

    The first half tests the model on turn 7 and on turn 9 in separate runs,
    with team 0 as the model team. The second half tests it on turn 8, with
    team 1 as the model team. Greedy controls every other turn in each model
    rollout. Each result is compared with all-greedy play on the same states.
    """
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty")
    if num_states < 2 or num_states % 2:
        raise ValueError("num_states must be a positive even number")
    if states is None:
        states = seq.data_generation.generate_random_sheet_state_for_turn(
            turn=6, num_sims=num_states, rng=np.random.default_rng(seed)
        )
    if states.x.shape[0] != num_states:
        raise ValueError("states must contain num_states rows")

    midpoint = num_states // 2
    first_half = state.take_sheet_states(states, np.arange(midpoint))
    second_half = state.take_sheet_states(states, np.arange(midpoint, num_states))
    greedy_scores = {
        0: _evaluate_policy_group(
            first_half,
            model=None,
            normalizer=None,
            model_turns=frozenset(),
            model_team=0,
            grid_size=grid_size,
            max_stones=max_stones,
        ),
        1: _evaluate_policy_group(
            second_half,
            model=None,
            normalizer=None,
            model_turns=frozenset(),
            model_team=1,
            grid_size=grid_size,
            max_stones=max_stones,
        ),
    }
    groups = (
        ("model_turn_7", frozenset((7,)), 0, first_half),
        ("model_turn_9", frozenset((9,)), 0, first_half),
        ("model_turn_8", frozenset((8,)), 1, second_half),
    )

    records = []
    for run_name, checkpoint_path in checkpoint_paths.items():
        model, normalizer = seq.load_model(checkpoint_path)
        for group, model_turns, model_team, group_states in groups:
            model_scores = _evaluate_policy_group(
                group_states,
                model,
                normalizer,
                model_turns=model_turns,
                model_team=model_team,
                grid_size=grid_size,
                max_stones=max_stones,
            )
            baseline = greedy_scores[model_team]
            differences = model_scores - baseline
            standard_error = float(
                np.std(differences, ddof=1) / np.sqrt(differences.size)
            )
            average_advantage = float(np.mean(differences))
            records.append({
                "run": run_name,
                "group": group,
                "model_team": model_team,
                "num_states": int(differences.size),
                "average_model_score": float(np.mean(model_scores)),
                "average_greedy_score": float(np.mean(baseline)),
                "average_advantage": average_advantage,
                "advantage_standard_error": standard_error,
                "advantage_ci95_low": average_advantage - 1.96 * standard_error,
                "advantage_ci95_high": average_advantage + 1.96 * standard_error,
                "win_rate": float(np.mean(differences > 0)),
                "tie_rate": float(np.mean(differences == 0)),
                "loss_rate": float(np.mean(differences < 0)),
            })
    return records
