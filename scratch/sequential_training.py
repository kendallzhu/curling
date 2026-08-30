"""Sequential final-score models for curling experiments.

The public functions in this module deliberately keep notebooks thin.  Models
predict team-0's final score differential, while throw selection changes the
sign when scoring from team 1's perspective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import hashlib
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

import curling_nn
import data_generation
import dataset
import nn
import physics
import scoring
import state
import stats
from bot import RandomThrows, ThrowSearcher, ThrowsGridSearcher


RandomSheetStates = Callable[..., state.SheetStates]
DatasetProgressCallback = Callable[["SequentialDataset", int], None]


def _batch_key(sheet: state.SheetStates, throw: state.Throws) -> str:
    """Create an exact, shape-aware key for one vectorized request."""
    digest = hashlib.sha256()
    for values in (
        sheet.first_team, sheet.x, sheet.y, sheet.velocities.v,
        sheet.velocities.theta, sheet.rotation_directions,
        throw.angle_deg, throw.speed, throw.turn, throw.y_val, throw.team,
    ):
        array = np.asarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class PhysicsCache:
    """Persistent whole-batch cache for vectorized physics calls."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.physics_simulations = 0

    def clear_stats(self) -> None:
        self.hits = self.misses = self.physics_simulations = 0

    def stats(self) -> dict[str, float | int]:
        queries = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "physics_simulations": self.physics_simulations,
            "hit_rate": self.hits / queries if queries else 0.0,
        }

    def _path_for(self, key: str) -> Path:
        return self.path / f"{key}.npz"

    def get(self, key: str):
        cache_path = self._path_for(key)
        if not cache_path.exists():
            return None
        with np.load(cache_path) as values:
            return state.SheetStates(
                first_team=np.array(values["first_team"], copy=True),
                x=np.array(values["x"], copy=True),
                y=np.array(values["y"], copy=True),
                velocities=state.Velocities(
                    v=np.array(values["velocity_v"], copy=True),
                    theta=np.array(values["velocity_theta"], copy=True),
                ),
                rotation_directions=np.array(values["rotation_directions"], copy=True),
            )

    def put(self, key: str, value: state.SheetStates) -> None:
        destination = self._path_for(key)
        temporary = self.path / f".{key}.tmp.npz"
        np.savez_compressed(
            temporary,
            first_team=value.first_team,
            x=value.x,
            y=value.y,
            velocity_v=value.velocities.v,
            velocity_theta=value.velocities.theta,
            rotation_directions=value.rotation_directions,
        )
        temporary.replace(destination)


def _default_cache_path() -> Path:
    return Path(".sequential_training_physics_cache")


GLOBAL_PHYSICS_CACHE = PhysicsCache(_default_cache_path())


class CachedPhysics:
    """Physics facade with the same vectorized interface plus cache lookup."""

    def __init__(self, backend=physics, cache: PhysicsCache = GLOBAL_PHYSICS_CACHE):
        self.backend = backend
        self.cache = cache

    def run_until_stopping(
        self, *, sheet_states: state.SheetStates, throws: state.Throws | None = None
    ) -> state.SheetStates:
        if throws is None:
            # Retain compatibility for callers that already added the stone.
            base_states = sheet_states
            throws = state.Throws(
                angle_deg=np.zeros(len(sheet_states.x)),
                speed=np.zeros(len(sheet_states.x)),
                turn=np.zeros(len(sheet_states.x), dtype=int),
                y_val=np.zeros(len(sheet_states.x)),
                team=np.zeros(len(sheet_states.x), dtype=int),
            )
        else:
            base_states = sheet_states
            sheet_states = state.add_stones_from_throws(sheet_states, throws)

        count = sheet_states.x.shape[0]
        if count == 0:
            return self.backend.run_until_stopping(sheet_states=sheet_states)
        key = _batch_key(base_states, throws)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.hits += 1
            return cached
        self.cache.misses += 1
        result = self.backend.run_until_stopping(sheet_states=sheet_states)
        self.cache.physics_simulations += count
        self.cache.put(key, result)
        return result

    def run_to_next_collision_or_stop(self, *, sheet_states: state.SheetStates):
        """Expose the other physics-facade operation for drop-in use."""
        return self.backend.run_to_next_collision_or_stop(sheet_states=sheet_states)


cached_physics = CachedPhysics()


def _legacy_random_sheet_states(
    *, turn: int, num_sims: int, rng: np.random.Generator
) -> state.SheetStates:
    return data_generation.random_sheet_states(
        team1=(turn + 1) // 2, team2=turn // 2, num_sims=num_sims, rng=rng
    )


@dataclass(frozen=True)
class ExperimentSetup:
    random_sheet_states: RandomSheetStates
    searcher: ThrowSearcher
    model_comparison_searcher: ThrowSearcher
    greedy_comparison_searcher: ThrowSearcher


def show_state_generator_versions() -> None:
    """Print the available training-state distributions and their meanings."""
    print("Available state generator versions:")
    print("  legacy_full: place exactly the expected number of live stones")
    print("               for the requested turn (the original behavior).")
    print("  turn_based:  simulate alternating throws, including stones")
    print("               removed by earlier throws (more realistic boards).")


def make_experiment_setup(state_generator_version: str, *, seed: int = 0) -> ExperimentSetup:
    """Build the standard generators and searchers for an experiment.

    ``state_generator_version`` selects the distribution used for boards at
    the beginning of each generated training example:

    * ``"legacy_full"``: place exactly the expected number of live stones
      for the requested turn.  This is the original behavior and is useful
      for reproducing older experiments.
    * ``"turn_based"``: simulate the requested number of alternating throws,
      including the possibility that earlier stones were removed.  The
      resulting board can therefore contain fewer live stones than throws
      that have occurred, making it a more realistic mid-end distribution.

    The value is intentionally explicit so experiment configuration and
    results can record which state distribution was used.
    """
    np.random.seed(seed)
    if state_generator_version == "legacy_full":
        random_sheet_states = _legacy_random_sheet_states
    elif state_generator_version == "turn_based":
        random_sheet_states = data_generation.generate_random_sheet_state_for_turn
    else:
        raise ValueError(
            "state_generator_version must be 'legacy_full' or 'turn_based'"
        )
    rng = np.random.default_rng(seed)
    return ExperimentSetup(
        random_sheet_states=random_sheet_states,
        searcher=make_default_throw_searcher(seed),
        model_comparison_searcher=ThrowsGridSearcher(10, 10, 4),
        greedy_comparison_searcher=GridAndRandomThrowSearcher(
            rng, grid_size=(10, 10, 4)
        ),
    )


def feature_width(max_stones: int) -> int:
    return 1 + 6 * max_stones


def raw_padded_sheet_state_features(
    sheet_states: state.SheetStates, max_stones: int
) -> np.ndarray:
    """Return fixed-width features for states containing 0..max_stones stones."""
    n_sims, n_stones = sheet_states.x.shape
    if n_stones > max_stones:
        raise ValueError(f"state has {n_stones} stones, maximum is {max_stones}")
    x = np.zeros((n_sims, max_stones), dtype=float)
    y = np.zeros_like(x)
    distance = np.zeros_like(x)
    in_house = np.zeros_like(x)
    x[:, :n_stones] = sheet_states.x
    y[:, :n_stones] = sheet_states.y
    if n_stones:
        distance[:, :n_stones] = sheet_states.distance_from_center_of_house()
        in_house[:, :n_stones] = (
            distance[:, :n_stones] < curling_nn.house_outer_circle_radius + curling_nn.STONE_RADIUS_M
        )
    thrown = np.zeros_like(x)
    thrown[:, :n_stones] = 1.0
    not_thrown = 1.0 - thrown
    return np.concatenate(
        [sheet_states.first_team[:, None], not_thrown, thrown, x, y, distance, in_house],
        axis=1,
    )


def padded_training_data(
    sheet_states: state.SheetStates,
    final_scores: np.ndarray,
    max_stones: int,
    num_stones_per_side: int,
) -> dataset.TrainingData:
    raw = raw_padded_sheet_state_features(sheet_states, max_stones)
    scores = np.asarray(final_scores, dtype=int).reshape(-1)
    if scores.size != raw.shape[0] or np.any(np.abs(scores) > num_stones_per_side):
        raise ValueError("score labels do not match the dataset")
    answers = (scores[:, None] == np.arange(-num_stones_per_side, num_stones_per_side + 1)).astype(float)
    normalizer = dataset.Normalizer.from_features(raw)
    return dataset.TrainingData(normalizer.normalize(raw), answers, normalizer, raw)


@dataclass
class SequentialDataset:
    sheet_states: state.SheetStates
    final_scores: np.ndarray
    max_stones: int
    num_stones_per_side: int

    def training_data(self) -> dataset.TrainingData:
        return padded_training_data(
            self.sheet_states, self.final_scores, self.max_stones, self.num_stones_per_side
        )


def fixed_validation_splits(
    data: SequentialDataset,
    *,
    train_sizes: tuple[int, ...],
    validation_size: int,
    seed: int = 0,
) -> dict[int, tuple[dataset.TrainingData, dataset.TrainingData]]:
    """Make nested training subsets sharing one held-out validation set."""
    if not train_sizes or any(size < 1 for size in train_sizes):
        raise ValueError("train_sizes must contain positive sizes")
    if validation_size < 1 or max(train_sizes) + validation_size > data.final_scores.size:
        raise ValueError("dataset is too small for the requested train/validation split")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(data.final_scores.size)
    validation_idx = indices[:validation_size]
    training_idx = indices[validation_size : validation_size + max(train_sizes)]
    validation = padded_training_data(
        state.take_sheet_states(data.sheet_states, validation_idx),
        data.final_scores[validation_idx], data.max_stones, data.num_stones_per_side,
    )
    splits = {}
    for train_size in train_sizes:
        train = padded_training_data(
            state.take_sheet_states(data.sheet_states, training_idx[:train_size]),
            data.final_scores[training_idx[:train_size]],
            data.max_stones, data.num_stones_per_side,
        )
        splits[train_size] = (train, validation)
    return splits


def write_sequential_dataset(path: str | Path, data: SequentialDataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = data.sheet_states
    np.savez_compressed(
        path,
        first_team=s.first_team,
        x=s.x,
        y=s.y,
        velocity_v=s.velocities.v,
        velocity_theta=s.velocities.theta,
        rotation_directions=s.rotation_directions,
        final_scores=np.asarray(data.final_scores),
        max_stones=np.asarray(data.max_stones),
        num_stones_per_side=np.asarray(data.num_stones_per_side),
    )


def load_sequential_dataset(path: str | Path) -> SequentialDataset:
    with np.load(path) as z:
        return SequentialDataset(
            state.SheetStates(
                first_team=np.array(z["first_team"], copy=True),
                x=np.array(z["x"], copy=True), y=np.array(z["y"], copy=True),
                velocities=state.Velocities(
                    v=np.array(z["velocity_v"], copy=True),
                    theta=np.array(z["velocity_theta"], copy=True),
                ),
                rotation_directions=np.array(z["rotation_directions"], copy=True),
            ),
            np.array(z["final_scores"], copy=True),
            int(z["max_stones"]), int(z["num_stones_per_side"]),
        )


def make_model(*, seed: int, max_stones: int, num_stones_per_side: int, hidden_layer_size: int = 20):
    return curling_nn.ScoreNetwork(
        seed=seed, num_stones=max_stones, input_layer_size=feature_width(max_stones),
        hidden_layer_size=hidden_layer_size, num_stones_per_side=num_stones_per_side,
    )


def probabilities(model, normalizer: dataset.Normalizer, sheet_states: state.SheetStates, max_stones: int) -> np.ndarray:
    features = normalizer.normalize(raw_padded_sheet_state_features(sheet_states, max_stones))
    return nn.softmax(model.run(features[:, :, None]))[:, :, 0]


def train_model(
    data: dataset.TrainingData, *, seed: int, max_stones: int, num_stones_per_side: int,
    hidden_layer_size: int = 20, batch_size: int = 100, num_iters: int = 30,
) -> tuple[object, dataset.Normalizer, dict[str, list[float]], dataset.TrainingData]:
    model = make_model(seed=seed, max_stones=max_stones, num_stones_per_side=num_stones_per_side, hidden_layer_size=hidden_layer_size)
    loss = nn.SoftmaxCrossEntropyLoss()
    train, validation = data.partition(0.2, seed=seed)
    train_losses, validation_losses = [], []
    for i in range(num_iters):
        learning_rate = 0.5 * (1 + np.cos(np.pi * i / num_iters))
        for batch in train.shuffle_batches(batch_size, seed=seed + i):
            model.train_batched(batch, loss, learning_rate, 0)
        train_losses.append(float(model.get_average_loss_batched(train.input_features, train.answers, loss)))
        validation_losses.append(float(model.get_average_loss_batched(validation.input_features, validation.answers, loss)))
    return (
        model,
        train.normalizer,
        {"train_loss": train_losses, "validation_loss": validation_losses},
        validation,
    )


def train_model_with_validation(
    train_data: dataset.TrainingData,
    validation_data: dataset.TrainingData,
    *,
    seed: int,
    max_stones: int,
    num_stones_per_side: int,
    hidden_layer_size: int = 20,
    batch_size: int = 100,
    num_iters: int = 30,
) -> tuple[object, dataset.Normalizer, dict[str, list[float]], dataset.TrainingData]:
    """Train on ``train_data`` while evaluating on a fixed validation set.

    The validation inputs are normalized with the training-set normalizer.
    This is useful for scaling experiments where every model must be compared
    on exactly the same held-out rows.
    """
    if train_data.size() < 1 or validation_data.size() < 1:
        raise ValueError("train and validation data must both be non-empty")
    model = make_model(
        seed=seed, max_stones=max_stones, num_stones_per_side=num_stones_per_side,
        hidden_layer_size=hidden_layer_size,
    )
    normalizer = train_data.normalizer
    validation = dataset.TrainingData(
        input_features=normalizer.normalize(validation_data.raw_inputs),
        answers=validation_data.answers,
        normalizer=normalizer,
        raw_inputs=validation_data.raw_inputs,
    )
    loss = nn.SoftmaxCrossEntropyLoss()
    train_losses, validation_losses = [], []
    for i in range(num_iters):
        learning_rate = 0.5 * (1 + np.cos(np.pi * i / num_iters))
        for batch in train_data.shuffle_batches(batch_size, seed=seed + i):
            model.train_batched(batch, loss, learning_rate, 0)
        train_losses.append(float(model.get_average_loss_batched(train_data.input_features, train_data.answers, loss)))
        validation_losses.append(float(model.get_average_loss_batched(validation.input_features, validation.answers, loss)))
    return model, normalizer, {"train_loss": train_losses, "validation_loss": validation_losses}, validation


class GridAndRandomThrowSearcher:
    """Candidate searcher matching the scaling experiments: 768 grid + 432 random."""

    def __init__(
        self, rng: np.random.Generator | None = None, grid_size=(8, 8, 4),
        random_count=36 * 4 * 3, *, seed: int | None = None,
    ):
        if rng is not None and seed is not None:
            raise ValueError("provide rng or seed, not both")
        rng = rng if rng is not None else np.random.default_rng(seed)
        self.grid_searcher = ThrowsGridSearcher(*grid_size)
        self.random_searcher = RandomThrows(rng, random_count)

    def get_throws(self, team: int):
        return state.concat_throws([
            self.grid_searcher.get_throws(team), self.random_searcher.get_throws(team)
        ])

    def get_throws_for_num_sims(self, *, team: int, sheet_states: state.SheetStates):
        grid_throws, grid_states = self.grid_searcher.get_throws_for_num_sims(
            team=team, sheet_states=sheet_states
        )
        random_throws, random_states = self.random_searcher.get_throws_for_num_sims(
            team=team, sheet_states=sheet_states
        )
        return state.concat_throws([grid_throws, random_throws]), state.concat(
            [grid_states, random_states]
        )


def make_default_throw_searcher(seed: int = 0) -> GridAndRandomThrowSearcher:
    return GridAndRandomThrowSearcher(np.random.default_rng(seed))


def evaluate_model(
    model, normalizer, data: dataset.TrainingData, *, N: int,
    num_bootstrap_samples: int = 1000, seed: int = 0,
) -> object:
    """Return the project's standard categorical prediction statistics."""
    evaluation_data = dataset.TrainingData(
        input_features=normalizer.normalize(data.raw_inputs),
        answers=data.answers,
        normalizer=normalizer,
        raw_inputs=data.raw_inputs,
    )
    predictions = stats.create_prediction_dataframe(
        model, evaluation_data, score_values=np.arange(-N, N + 1)
    )
    return stats.compute_stats(
        predictions, num_bootstrap_samples=num_bootstrap_samples, seed=seed
    )


def save_model_evaluation(
    path: str | Path, *, model, normalizer, data: dataset.TrainingData,
    training_info: dict, model_index: int, N: int,
    num_bootstrap_samples: int = 1000, seed: int = 0,
) -> dict:
    """Evaluate a checkpoint and save JSON that can be read during training."""
    evaluation = evaluate_model(
        model, normalizer, data, N=N,
        num_bootstrap_samples=num_bootstrap_samples, seed=seed,
    )
    record = {
        "model_index": model_index,
        "dataset_size": data.size(),
        "training": training_info,
        "stats": asdict(evaluation),
    }
    save_metadata(path, record)
    return record


def write_model(path: str | Path, model, normalizer: dataset.Normalizer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "num_stones": np.asarray(model.num_stones),
        "hidden_layer_size": np.asarray(model.hidden_layer_size),
        "num_stones_per_side": np.asarray(model.num_stones_per_side),
        "feature_means": normalizer.feature_means,
        "feature_stdevs": normalizer.feature_stdevs,
    }
    for i, layer in enumerate(model.linear_layers()):
        arrays[f"w{i}"] = layer.weights
        arrays[f"b{i}"] = layer.bias
    np.savez(path, **arrays)


def load_model(path: str | Path):
    with np.load(path) as z:
        model = make_model(
            seed=0,
            max_stones=int(z["num_stones"]),
            num_stones_per_side=int(z["num_stones_per_side"]),
            hidden_layer_size=int(z["hidden_layer_size"]),
        )
        for i, layer in enumerate(model.linear_layers()):
            layer.weights = np.array(z[f"w{i}"], copy=True)
            layer.bias = np.array(z[f"b{i}"], copy=True)
        normalizer = dataset.Normalizer(
            feature_means=np.array(z["feature_means"], copy=True),
            feature_stdevs=np.array(z["feature_stdevs"], copy=True),
        )
    return model, normalizer


def policy_summary(comparison: PolicyComparison) -> dict[str, float]:
    """Compute aggregate metrics comparing the model and greedy policies.

    Win, tie, and loss probabilities describe the paired policy comparison,
    rather than whether team 0 won the game.  They are measured from the
    perspective of the team making the compared throw.
    """
    model = comparison.model_scores
    greedy = comparison.greedy_scores
    if comparison.initial_states is None:
        # Preserve the simple synthetic-test behavior for comparisons without
        # board states. Real comparisons always provide initial_states.
        advantage = model - greedy
    else:
        acting_team = comparison.initial_states.next_team_to_play()
        advantage = np.where(acting_team == 0, model - greedy, greedy - model)
    score_difference = model - greedy
    score_difference_stderr = float(
        np.std(score_difference, ddof=1) / np.sqrt(score_difference.size)
    ) if score_difference.size > 1 else float("nan")
    return {
        "model_expected_score": float(np.mean(model)),
        "greedy_expected_score": float(np.mean(greedy)),
        "expected_score_difference": float(np.mean(model - greedy)),
        "expected_score_difference_stderr": score_difference_stderr,
        "model_win_probability": float(np.mean(advantage > 0)),
        "model_tie_probability": float(np.mean(advantage == 0)),
        "model_loss_probability": float(np.mean(advantage < 0)),
        "greedy_win_probability": float(np.mean(advantage < 0)),
        "greedy_tie_probability": float(np.mean(advantage == 0)),
        "greedy_loss_probability": float(np.mean(advantage > 0)),
    }


def _candidate_results(sheet_states, throws, tiled_states):
    return cached_physics.run_until_stopping(sheet_states=tiled_states, throws=throws)


def _choose_throw(sheet_states, team, searcher, scorer):
    throws, tiled = searcher.get_throws_for_num_sims(team=team, sheet_states=sheet_states)
    n = sheet_states.x.shape[0]
    if n == 0:
        return state.Throws(*(np.array([]) for _ in range(5)))
    result = _candidate_results(sheet_states, throws, tiled)
    scores = np.asarray(scorer(result, team))
    candidates = scores.reshape(-1, n).argmax(axis=0) * n + np.arange(n)
    return state.Throws(*(getattr(throws, name)[candidates] for name in ("angle_deg", "speed", "turn", "y_val", "team")))


def choose_model_throws(sheet_states, model, normalizer, max_stones, searcher, team):
    def score(result, team):
        return np.where(team == 0, 1, -1) * model.expected_score(
            model.run(normalizer.normalize(raw_padded_sheet_state_features(result, max_stones))[:, :, None])
        )
    return _choose_throw(sheet_states, team, searcher, score)


def choose_greedy_throws(sheet_states, searcher, team):
    return _choose_throw(sheet_states, team, searcher, lambda result, team: scoring.get_net_score_for_team(result, team))


def model_candidate_diagnostics(
    sheet_states, model, normalizer, *, max_stones: int, searcher, team: int
) -> dict[str, float]:
    """Compare predictions with immediate outcomes for every candidate."""
    throws, tiled = searcher.get_throws_for_num_sims(
        team=team, sheet_states=sheet_states
    )
    n = sheet_states.x.shape[0]
    result = _candidate_results(sheet_states, throws, tiled)
    predictions = model.expected_score(
        model.run(
            normalizer.normalize(
                raw_padded_sheet_state_features(result, max_stones)
            )[:, :, None]
        )
    ).reshape(-1, n)
    # Keep predictions and realized values in the same team-0 perspective.
    actual = scoring.get_net_score_for_team(result, 0).reshape(-1, n)
    model_utility = predictions if team == 0 else -predictions
    actual_utility = actual if team == 0 else -actual
    model_indices = model_utility.argmax(axis=0)
    greedy_indices = actual_utility.argmax(axis=0)
    correlation = np.corrcoef(predictions.reshape(-1), actual.reshape(-1))[0, 1]
    return {
        "selected_predicted_score": float(predictions[model_indices, np.arange(n)].mean()),
        "greedy_predicted_score": float(predictions[greedy_indices, np.arange(n)].mean()),
        "selected_immediate_score": float(actual[model_indices, np.arange(n)].mean()),
        "greedy_immediate_score": float(actual[greedy_indices, np.arange(n)].mean()),
        "candidate_prediction_actual_correlation": float(correlation),
    }


def rollout_to_terminal(sheet_states, models: Mapping[int, tuple[object, dataset.Normalizer]], max_stones: int, searcher: ThrowSearcher) -> tuple[state.SheetStates, np.ndarray]:
    current = sheet_states
    while current.x.shape[1] < max_stones:
        n = current.x.shape[1]
        team = int(current.next_team_to_play()[0])
        if n == max_stones - 1:
            throws = choose_greedy_throws(current, searcher, team)
        else:
            model, normalizer = models[n + 1]
            throws = choose_model_throws(current, model, normalizer, max_stones, searcher, team)
        current = cached_physics.run_until_stopping(sheet_states=current, throws=throws)
    return current, scoring.get_net_score_for_team(current, 0)


def generate_dataset(
    i: int, *, num_rows: int = 10_000, N: int = 5, seed: int = 0,
    models=None, searcher=None, batch_size: int = 32,
    shard_dir: str | Path | None = None, shard_size: int = 500,
    random_sheet_states: RandomSheetStates | None = None,
    diagnostic_callback: DatasetProgressCallback | None = None,
    diagnostic_train_sizes: tuple[int, ...] = (),
    diagnostic_evaluation_size: int = 1000,
) -> SequentialDataset:
    np.random.seed(seed)
    max_stones = 2 * N
    if not 1 <= i < max_stones:
        raise ValueError("i must be between 1 and 2N-1")
    if models is None and i < max_stones - 1:
        raise ValueError("models m_(i+1)..m_(2N-1) are required")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if diagnostic_evaluation_size < 1:
        raise ValueError("diagnostic_evaluation_size must be positive")
    if any(size < 1 for size in diagnostic_train_sizes):
        raise ValueError("diagnostic_train_sizes must contain positive sizes")
    if tuple(sorted(set(diagnostic_train_sizes))) != diagnostic_train_sizes:
        raise ValueError("diagnostic_train_sizes must be sorted and contain no duplicates")
    rng = np.random.default_rng(seed)
    searcher = searcher or GridAndRandomThrowSearcher(rng)
    if isinstance(searcher, GridAndRandomThrowSearcher):
        searcher.random_searcher.rng = rng
    random_sheet_states = random_sheet_states or _legacy_random_sheet_states
    parts_states, parts_scores = [], []
    pending_states, pending_scores = [], []
    next_shard = 0
    next_diagnostic = 0

    def flush_shard(*, final: bool = False) -> None:
        nonlocal pending_states, pending_scores, next_shard
        if shard_dir is None or not pending_states:
            return
        combined_states = state.concat(pending_states)
        combined_scores = np.concatenate(pending_scores)
        pending_states, pending_scores = [], []
        while combined_states.x.shape[0] >= shard_size:
            shard_states = state.take_sheet_states(combined_states, np.arange(shard_size))
            write_sequential_dataset(
                Path(shard_dir) / f"D_{i}_batch{next_shard:05d}.npz",
                SequentialDataset(shard_states, combined_scores[:shard_size], max_stones, N),
            )
            next_shard += 1
            combined_states = state.take_sheet_states(
                combined_states, np.arange(shard_size, combined_states.x.shape[0])
            )
            combined_scores = combined_scores[shard_size:]
        if combined_states.x.shape[0] and final:
            write_sequential_dataset(
                Path(shard_dir) / f"D_{i}_batch{next_shard:05d}.npz",
                SequentialDataset(combined_states, combined_scores, max_stones, N),
            )
            next_shard += 1
        elif combined_states.x.shape[0]:
            pending_states.append(combined_states)
            pending_scores.append(combined_scores)
    team0 = (i + 1) // 2
    team1 = i // 2
    for start in range(0, num_rows, batch_size):
        count = min(batch_size, num_rows - start)
        initial = random_sheet_states(turn=i, num_sims=count, rng=rng)
        final, scores = rollout_to_terminal(initial, models or {}, max_stones, searcher)
        parts_states.append(initial)
        parts_scores.append(scores)
        pending_states.append(initial)
        pending_scores.append(scores)
        if sum(part.x.shape[0] for part in pending_states) >= shard_size:
            flush_shard()
        if diagnostic_callback is not None:
            generated_rows = sum(part.x.shape[0] for part in parts_states)
            while (
                next_diagnostic < len(diagnostic_train_sizes)
                and generated_rows
                >= diagnostic_evaluation_size + diagnostic_train_sizes[next_diagnostic]
            ):
                train_size = diagnostic_train_sizes[next_diagnostic]
                diagnostic_size = diagnostic_evaluation_size + train_size
                generated_states = state.concat(parts_states)
                diagnostic_callback(
                    SequentialDataset(
                        state.take_sheet_states(
                            generated_states, np.arange(diagnostic_size)
                        ),
                        np.concatenate(parts_scores)[:diagnostic_size],
                        max_stones,
                        N,
                    ),
                    train_size,
                )
                next_diagnostic += 1
    flush_shard(final=True)
    if diagnostic_callback is not None and next_diagnostic != len(diagnostic_train_sizes):
        raise ValueError("diagnostic_train_sizes cannot exceed num_rows")
    return SequentialDataset(state.concat(parts_states), np.concatenate(parts_scores), max_stones, N)


@dataclass
class PolicyComparison:
    initial_states: state.SheetStates
    model_scores: np.ndarray
    greedy_scores: np.ndarray
    model_throws: state.Throws
    greedy_throws: state.Throws


def write_policy_comparison(path: str | Path, comparison: PolicyComparison) -> None:
    """Persist all rows needed to recompute policy metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = comparison.initial_states
    arrays = {
        "first_team": s.first_team,
        "x": s.x,
        "y": s.y,
        "velocity_v": s.velocities.v,
        "velocity_theta": s.velocities.theta,
        "rotation_directions": s.rotation_directions,
        "model_scores": comparison.model_scores,
        "greedy_scores": comparison.greedy_scores,
    }
    for prefix, throws in (("model", comparison.model_throws), ("greedy", comparison.greedy_throws)):
        for name in ("angle_deg", "speed", "turn", "y_val", "team"):
            arrays[f"{prefix}_{name}"] = getattr(throws, name)
    np.savez_compressed(path, **arrays)


def compare_policies(
    i: int,
    sheet_states: state.SheetStates,
    models,
    *,
    max_stones: int,
    searcher=None,
    model_searcher=None,
) -> PolicyComparison:
    if sheet_states.x.shape[1] != i - 1:
        raise ValueError("comparison states must contain i-1 stones")
    searcher = searcher or make_default_throw_searcher()
    model_searcher = model_searcher or searcher
    team = int(sheet_states.next_team_to_play()[0])
    model, normalizer = models[i]
    model_throw = choose_model_throws(
        sheet_states, model, normalizer, max_stones, model_searcher, team
    )
    greedy_throw = choose_greedy_throws(sheet_states, searcher, team)
    model_after, model_scores = rollout_to_terminal(
        cached_physics.run_until_stopping(sheet_states=sheet_states, throws=model_throw),
        models, max_stones, searcher
    )
    greedy_after, greedy_scores = rollout_to_terminal(
        cached_physics.run_until_stopping(sheet_states=sheet_states, throws=greedy_throw),
        models, max_stones, searcher
    )
    return PolicyComparison(sheet_states, model_scores, greedy_scores, model_throw, greedy_throw)


def print_policy_comparison(
    i: int,
    *,
    artifact_dir: str | Path,
    models,
    random_sheet_states: RandomSheetStates,
    greedy_comparison_searcher: ThrowSearcher,
    model_comparison_searcher: ThrowSearcher,
    max_stones: int,
    num_sims: int = 5,
) -> PolicyComparison:
    """Compare and persist model and greedy policies for one model index."""
    if num_sims < 1:
        raise ValueError("num_sims must be positive")
    states = random_sheet_states(
        turn=i - 1, num_sims=num_sims, rng=np.random.default_rng(i)
    )
    comparison = compare_policies(
        i,
        states,
        models,
        max_stones=max_stones,
        searcher=greedy_comparison_searcher,
        model_searcher=model_comparison_searcher,
    )
    print(i, policy_summary(comparison))
    write_policy_comparison(
        Path(artifact_dir) / f"policy_comparison_{i}.npz", comparison
    )
    return comparison


def train_diagnostic_model(
    generated: SequentialDataset,
    *,
    train_size: int,
    evaluation_size: int = 1000,
    model_index: int,
    output_dir: str | Path,
    num_bootstrap_samples: int = 1000,
) -> dict:
    """Train and persist one model on a prefix after a fixed evaluation prefix."""
    if evaluation_size < 1 or train_size < 1:
        raise ValueError("evaluation_size and train_size must be positive")
    if evaluation_size + train_size > generated.final_scores.size:
        raise ValueError("generated data is too small for the diagnostic split")

    evaluation = padded_training_data(
        state.take_sheet_states(generated.sheet_states, np.arange(evaluation_size)),
        generated.final_scores[:evaluation_size],
        generated.max_stones,
        generated.num_stones_per_side,
    )
    training = padded_training_data(
        state.take_sheet_states(
            generated.sheet_states,
            np.arange(evaluation_size, evaluation_size + train_size),
        ),
        generated.final_scores[evaluation_size : evaluation_size + train_size],
        generated.max_stones,
        generated.num_stones_per_side,
    )
    model, normalizer, training_info, _ = train_model_with_validation(
        training,
        evaluation,
        seed=model_index,
        max_stones=generated.max_stones,
        num_stones_per_side=generated.num_stones_per_side,
    )
    model_stats = evaluate_model(
        model,
        normalizer,
        evaluation,
        N=generated.num_stones_per_side,
        num_bootstrap_samples=num_bootstrap_samples,
        seed=model_index,
    )
    record = {
        "model_index": model_index,
        "evaluation_size": evaluation_size,
        "train_size": train_size,
        "training_dataset_size": training.size(),
        "stats": {
            "r_squared": asdict(model_stats.r_squared),
            "negative_log_probability": asdict(model_stats.negative_log_probability),
        },
        "training": training_info,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_model(output_dir / f"m_{model_index}_{train_size}.npz", model, normalizer)
    with (output_dir / f"m_{model_index}_stats.jsonl").open("a") as stats_file:
        stats_file.write(json.dumps(record, sort_keys=True) + "\n")
    print(
        f"diagnostic m_{model_index} train={train_size}: "
        f"R²={model_stats.r_squared.value:.3f}, "
        f"-log P(actual)={model_stats.negative_log_probability.value:.3f}"
    )
    return record


def _default_diagnostic_train_sizes(rows_per_model: int) -> tuple[int, ...]:
    """Return 1000, 2000, 4000, ... sizes that fit after the eval prefix."""
    sizes = []
    size = 1000
    while 1000 + size <= rows_per_model:
        sizes.append(size)
        size *= 2
    return tuple(sizes)


@dataclass
class SequentialTrainingRun:
    models: dict
    datasets: dict[int, SequentialDataset]
    validation_datasets: dict[int, dataset.TrainingData]
    training_info: dict


def train_sequential_models(
    *, N: int, rows_per_model: int, artifact_dir: str | Path,
    resume_from: int | None, training_stages: int,
    experiment: ExperimentSetup,
    diagnostic_output_dir: str | Path | None = None,
    policy_comparison_num_sims: int = 5,
) -> SequentialTrainingRun:
    """Train, save, and evaluate the sequential model chain."""
    artifact_dir = Path(artifact_dir)
    max_stones = 2 * N
    if training_stages < 1 or training_stages > max_stones - 1:
        raise ValueError("training_stages must be between 1 and 2N-1")
    if policy_comparison_num_sims < 1:
        raise ValueError("policy_comparison_num_sims must be positive")
    models, datasets, validation_datasets, training_info = {}, {}, {}, {}
    first_model = max_stones - 1 if resume_from is None else resume_from
    last_model = max(1, first_model - training_stages + 1)

    if resume_from is not None:
        for i in range(resume_from + 1, max_stones):
            if i == max_stones - 1:
                generated = load_sequential_dataset(artifact_dir / f"D_{i}.npz")
                data = generated.training_data()
                model, normalizer, info, validation = train_model(
                    data, seed=i, max_stones=max_stones, num_stones_per_side=N
                )
                datasets[i], validation_datasets[i] = generated, validation
                models[i], training_info[i] = (model, normalizer), info
                write_model(artifact_dir / f"m_{i}.npz", model, normalizer)
                save_model_evaluation(
                    artifact_dir / f"m_{i}_evaluation.json", model=model,
                    normalizer=normalizer, data=validation,
                    training_info=info, model_index=i, N=N
                )
                print_policy_comparison(
                    i, artifact_dir=artifact_dir, models=models,
                    random_sheet_states=experiment.random_sheet_states,
                    greedy_comparison_searcher=experiment.greedy_comparison_searcher,
                    model_comparison_searcher=experiment.model_comparison_searcher,
                    max_stones=max_stones,
                    num_sims=policy_comparison_num_sims,
                )
            else:
                models[i] = load_model(artifact_dir / f"m_{i}.npz")
                evaluation_path = artifact_dir / f"m_{i}_evaluation.json"
                if evaluation_path.exists():
                    training_info[i] = json.loads(evaluation_path.read_text())

    for i in range(first_model, last_model - 1, -1):
        diagnostic_sizes = (
            _default_diagnostic_train_sizes(rows_per_model)
            if diagnostic_output_dir is not None
            else ()
        )

        def diagnostic_callback(generated, train_size, model_index=i):
            train_diagnostic_model(
                generated,
                train_size=train_size,
                model_index=model_index,
                output_dir=diagnostic_output_dir,
            )

        generated = generate_dataset(
            i, num_rows=rows_per_model, N=N, seed=i, models=models,
            searcher=experiment.searcher, shard_dir=artifact_dir / f"D_{i}",
            shard_size=500, random_sheet_states=experiment.random_sheet_states,
            diagnostic_callback=diagnostic_callback if diagnostic_sizes else None,
            diagnostic_train_sizes=diagnostic_sizes,
        )
        data = generated.training_data()
        model, normalizer, info, validation = train_model(
            data, seed=i, max_stones=max_stones, num_stones_per_side=N
        )
        datasets[i], validation_datasets[i] = generated, validation
        models[i], training_info[i] = (model, normalizer), info
        write_sequential_dataset(artifact_dir / f"D_{i}.npz", generated)
        write_model(artifact_dir / f"m_{i}.npz", model, normalizer)
        save_model_evaluation(
            artifact_dir / f"m_{i}_evaluation.json", model=model,
            normalizer=normalizer, data=validation,
            training_info=info, model_index=i, N=N
        )
        print_policy_comparison(
            i, artifact_dir=artifact_dir, models=models,
            random_sheet_states=experiment.random_sheet_states,
            greedy_comparison_searcher=experiment.greedy_comparison_searcher,
            model_comparison_searcher=experiment.model_comparison_searcher,
            max_stones=max_stones,
            num_sims=policy_comparison_num_sims,
        )
    print(f"trained {len(models)} models")
    return SequentialTrainingRun(models, datasets, validation_datasets, training_info)


def generate_final_dataset(models, *, k: int, fractions: Mapping[int, float], num_rows: int = 10_000, N: int = 5, seed: int = 0, random_sheet_states: RandomSheetStates | None = None) -> dataset.TrainingData:
    np.random.seed(seed)
    max_stones = 2 * N
    counts = list(range(k, max_stones))
    if set(fractions) != set(counts) or any(v < 0 for v in fractions.values()) or not np.isclose(sum(fractions.values()), 1):
        raise ValueError("fractions must cover k..2N-1 and sum to one")
    raw_parts, answer_parts = [], []
    rng = np.random.default_rng(seed)
    random_sheet_states = random_sheet_states or _legacy_random_sheet_states
    raw_counts = np.floor(np.asarray([fractions[n] for n in counts]) * num_rows).astype(int)
    raw_counts[-1] += num_rows - int(raw_counts.sum())
    for n, count in zip(counts, raw_counts):
        states = random_sheet_states(turn=n, num_sims=int(count), rng=rng)
        model, normalizer = models[n]
        raw_parts.append(raw_padded_sheet_state_features(states, max_stones))
        answer_parts.append(probabilities(model, normalizer, states, max_stones))
    raw = np.concatenate(raw_parts)
    answers = np.concatenate(answer_parts)
    normalizer = dataset.Normalizer.from_features(raw)
    return dataset.TrainingData(normalizer.normalize(raw), answers, normalizer, raw)


def save_metadata(path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
