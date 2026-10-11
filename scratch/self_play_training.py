"""Iterative self-play training for a multi-turn score network.

The notebook next to this module is intentionally just configuration and
display code.  This module owns the vectorized rollout, dataset retention,
checkpointing, and training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import dataset
import nn
import scoring
import state
from bot import ThrowPerturbations, ThrowsGridSearcher, select_robust_throws
from scratch import sequential_training as seq


@dataclass(frozen=True)
class SelfPlayConfig:
    num_states: int = 500
    num_iterations: int = 30
    grid_size: tuple[int, int, int] = (10, 10, 4)
    state_generator_version: str = "turn_based"
    halflife: float = 5.0
    max_stones: int = 10
    num_stones_per_side: int = 5
    hidden_layer_size: int = 20
    batch_size: int = 100
    training_iterations: int = 30
    learning_rate: float = 0.5
    validation_fraction: float = 0.0
    robust_top_fraction: float = 0.05
    perturbations: ThrowPerturbations = ThrowPerturbations(
        angle_offsets_deg=(0.0, -0.2, 0.2),
        probabilities=(0.5, 0.25, 0.25),
    )

    def retention_probability(self) -> float:
        if self.halflife <= 0:
            raise ValueError("halflife must be positive")
        return 2.0 ** (-1.0 / self.halflife)


def _pad_states(sheet_states: state.SheetStates, max_stones: int) -> state.SheetStates:
    """Pad variable-length rollout states to the network's fixed width."""
    n_sims, n_stones = sheet_states.x.shape
    if n_stones > max_stones:
        raise ValueError("state has more stones than max_stones")
    padding = max_stones - n_stones
    if padding == 0:
        return sheet_states
    zeros = np.zeros((n_sims, padding))
    return state.SheetStates(
        first_team=sheet_states.first_team.copy(),
        x=np.concatenate((sheet_states.x, zeros), axis=1),
        y=np.concatenate((sheet_states.y, zeros), axis=1),
        velocities=state.Velocities(
            v=np.concatenate((sheet_states.velocities.v, zeros), axis=1),
            theta=np.concatenate((sheet_states.velocities.theta, zeros), axis=1),
        ),
        rotation_directions=np.concatenate(
            (sheet_states.rotation_directions, np.zeros_like(zeros, dtype=int)), axis=1
        ),
    )


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
    """Generate D_i: three labelled states for each random s_7."""
    if config.state_generator_version != "turn_based":
        raise ValueError("self-play currently requires state_generator_version='turn_based'")
    generator = seq.data_generation.generate_random_sheet_state_for_turn
    s7 = generator(turn=7, num_sims=config.num_states, rng=rng)
    searcher = ThrowsGridSearcher(*config.grid_size)

    def model_utility(states):
        scores = _model_scores(model, normalizer, _pad_states(states, config.max_stones), config.max_stones)
        return scores if team == 0 else -scores

    team = int(s7.next_team_to_play()[0])
    s8_throw = _choose_throw(
        s7, team, searcher, model_utility, model_utility,
        config.perturbations, config.robust_top_fraction,
    )
    s8 = seq.cached_physics.run_until_stopping(sheet_states=s7, throws=s8_throw)

    team = int(s8.next_team_to_play()[0])
    s9_throw = _choose_throw(
        s8, team, searcher, model_utility, model_utility,
        config.perturbations, config.robust_top_fraction,
    )
    s9 = seq.cached_physics.run_until_stopping(sheet_states=s8, throws=s9_throw)

    team = int(s9.next_team_to_play()[0])
    def actual_utility(states):
        return scoring.get_net_score_for_team(states, team)

    s10_throw = _choose_throw(
        s9, team, searcher, actual_utility, actual_utility,
        config.perturbations, config.robust_top_fraction,
    )
    s10 = seq.cached_physics.run_until_stopping(sheet_states=s9, throws=s10_throw)
    final_scores = scoring.get_net_score_for_team(s10, 0)

    rows = state.concat([_pad_states(s7, config.max_stones), _pad_states(s8, config.max_stones), _pad_states(s9, config.max_stones)])
    labels = np.tile(final_scores, 3)
    return seq.SequentialDataset(rows, labels, config.max_stones, config.num_stones_per_side), s10


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
) -> tuple[object, dataset.Normalizer, list[float]]:
    """Train in-place from the supplied checkpoint using the old schedule."""
    training_data = data.training_data()
    loss = nn.SoftmaxCrossEntropyLoss()
    losses = []
    for iteration in range(config.training_iterations):
        learning_rate = config.learning_rate * (
            0.5 * (1.0 + np.cos(np.pi * iteration / config.training_iterations))
        )
        for batch in training_data.shuffle_batches(config.batch_size, seed=seed + iteration):
            model.train_batched(batch, loss, learning_rate, 0)
        losses.append(float(model.get_average_loss_batched(
            training_data.input_features, training_data.answers, loss
        )))
    return model, training_data.normalizer, losses


def run_self_play(
    *,
    config: SelfPlayConfig,
    artifact_dir: str | Path,
    initial_model_path: str | Path,
    initial_dataset_path: str | Path,
    seed: int = 0,
    resume_from: int | None = None,
) -> dict[int, dict[str, object]]:
    """Run and persist self-play iterations, returning checkpoint metadata."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if resume_from is None:
        model, normalizer = seq.load_model(initial_model_path)
        previous = seq.load_sequential_dataset(initial_dataset_path)
        start = 1
    else:
        model, normalizer = seq.load_model(artifact_dir / f"m_{resume_from}.npz")
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
        model, normalizer, losses = train_from_weights(model, combined, config=config, seed=seed + i)
        seq.write_sequential_dataset(artifact_dir / f"D_{i}.npz", generated)
        seq.write_sequential_dataset(artifact_dir / f"X_{i}.npz", combined)
        seq.write_model(artifact_dir / f"m_{i}.npz", model, normalizer)
        results[i] = {"new_rows": generated.final_scores.size, "training_rows": combined.final_scores.size, "loss": losses}
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
    current = initial_states
    searcher = ThrowsGridSearcher(*grid_size)
    for turn in range(7, 11):
        if turn in model_turns:
            throws = _greedy_model_throw(
                current, model, normalizer, searcher, max_stones
            )
        else:
            throws = _greedy_grid_throw(current, searcher)
        current = seq.cached_physics.run_until_stopping(
            sheet_states=current, throws=throws
        )
    return scoring.get_net_score_for_team(current, model_team)


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
        model, normalizer = seq.load_model(
            Path(checkpoint_dir) / f"m_{checkpoint}.npz"
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
