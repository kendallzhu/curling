"""Training and evaluation helpers for value-network scaling experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

import bot
import curling_nn
import dataset
import evaluation
import nn
import stats


TRAINING_SIZES = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000)
HIDDEN_LAYER_SIZES = (10, 15, 20, 30, 50)
MAX_TRAINING_SIZE = 300_000


def make_scaling_splits(
    data: dataset.TrainingData,
    *,
    validation_size: int = 30_000,
    max_training_size: int = MAX_TRAINING_SIZE,
    seed: int = 0,
) -> tuple[dataset.TrainingData, dataset.TrainingData]:
    """Return a deterministic validation set and a shuffled training pool.

    The returned training pool is large enough for the sweep. Its prefixes are
    the training sets used by the different dataset-size runs.
    """
    required_size = validation_size + max_training_size
    if data.size() < required_size:
        raise ValueError(
            f"need at least {required_size:,} rows, found {data.size():,}"
        )
    rng = np.random.default_rng(seed)
    indices = rng.permutation(data.size())[:required_size]
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    validation_raw = data.raw_inputs[validation_indices]
    validation_answers = data.answers[validation_indices]
    training_raw = data.raw_inputs[training_indices]
    training_answers = data.answers[training_indices]
    # These are raw rows. Each run fits its normalizer only on its own subset.
    raw_normalizer = dataset.Normalizer.from_features(training_raw)
    training_pool = dataset.TrainingData(
        input_features=raw_normalizer.normalize(training_raw),
        answers=training_answers,
        normalizer=raw_normalizer,
        raw_inputs=training_raw,
    )
    validation = dataset.TrainingData(
        input_features=raw_normalizer.normalize(validation_raw),
        answers=validation_answers,
        normalizer=raw_normalizer,
        raw_inputs=validation_raw,
    )
    return training_pool, validation


def _subset_with_training_normalizer(
    training_pool: dataset.TrainingData,
    size: int,
) -> tuple[dataset.TrainingData, dataset.Normalizer]:
    raw_inputs = training_pool.raw_inputs[:size]
    normalizer = dataset.Normalizer.from_features(raw_inputs)
    return dataset.TrainingData(
        input_features=normalizer.normalize(raw_inputs),
        answers=training_pool.answers[:size],
        normalizer=normalizer,
        raw_inputs=raw_inputs,
    ), normalizer


def train_value_network(
    training_data: dataset.TrainingData,
    *,
    hidden_layer_size: int,
    seed: int,
    num_epochs: int = 30,
    batch_size: int = 1_000,
    initial_learning_rate: float = 0.05,
) -> curling_nn.ValueNetwork:
    """Train one value network using the same cosine schedule as training.ipynb."""
    num_stones = (training_data.raw_inputs.shape[1] - 1) // 5
    num_stones_per_side = (training_data.answers.shape[1] - 1) // 2
    network = curling_nn.ValueNetwork(
        seed=seed,
        num_stones=num_stones,
        hidden_layer_size=hidden_layer_size,
        num_stones_per_side=num_stones_per_side,
    )
    loss_function = nn.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        learning_rate = initial_learning_rate * 0.5 * (
            1 + np.cos(np.pi * epoch / num_epochs)
        )
        for batch in training_data.shuffle_batches(batch_size, seed=seed + epoch):
            network.train_batched(batch, loss_function, learning_rate, 0)
    return network


def checkpoint_path(
    weights_dir: str | Path, training_size: int, hidden_layer_size: int
) -> Path:
    return Path(weights_dir) / f"value_n{training_size}_h{hidden_layer_size}.npz"


def train_sweep(
    training_pool: dataset.TrainingData,
    *,
    weights_dir: str | Path,
    training_sizes: Iterable[int] = TRAINING_SIZES,
    hidden_layer_sizes: Iterable[int] = HIDDEN_LAYER_SIZES,
    hidden_layer_sizes_by_training_size: dict[int, Iterable[int]] | None = None,
    seed: int = 0,
    num_epochs: int = 30,
    batch_size: int = 1_000,
    initial_learning_rate: float = 0.05,
) -> list[Path]:
    """Train and save every size/width combination, skipping existing files."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for training_size in training_sizes:
        training_data, normalizer = _subset_with_training_normalizer(
            training_pool, training_size
        )
        sizes = (
            hidden_layer_sizes_by_training_size.get(training_size, hidden_layer_sizes)
            if hidden_layer_sizes_by_training_size is not None
            else hidden_layer_sizes
        )
        for hidden_layer_size in sizes:
            path = checkpoint_path(weights_dir, training_size, hidden_layer_size)
            paths.append(path)
            if path.exists():
                continue
            run_seed = seed + training_size + 10_000 * hidden_layer_size
            network = train_value_network(
                training_data,
                hidden_layer_size=hidden_layer_size,
                seed=run_seed,
                num_epochs=num_epochs,
                batch_size=batch_size,
                initial_learning_rate=initial_learning_rate,
            )
            curling_nn.write_v_weights(path, network, normalizer)
    return paths


def evaluate_sweep(
    paths: Iterable[Path],
    validation_data: dataset.TrainingData,
    *,
    num_bootstrap_samples: int = 100,
) -> pl.DataFrame:
    """Evaluate checkpoints efficiently and return one row per checkpoint."""
    rows: list[dict[str, float | int | str]] = []
    for path in paths:
        path = Path(path)
        network, normalizer = curling_nn.load_v_weights(path)
        evaluation_data = dataset.TrainingData(
            input_features=normalizer.normalize(validation_data.raw_inputs),
            answers=validation_data.answers,
            normalizer=normalizer,
            raw_inputs=validation_data.raw_inputs,
        )
        predictions = stats.create_prediction_dataframe(network, evaluation_data)
        model_stats = stats.compute_stats(
            predictions, num_bootstrap_samples=num_bootstrap_samples
        )
        rows.append(
            {
                "path": str(path),
                "training_size": int(path.stem.split("_n")[1].split("_h")[0]),
                "hidden_layer_size": network.hidden_layer_size,
                "num_parameters": curling_nn.print_trainable_parameter_info(network),
                "r_squared": model_stats.r_squared.value,
                "negative_log_probability": model_stats.negative_log_probability.value,
            }
        )
    return pl.DataFrame(rows).sort(["training_size", "hidden_layer_size"])


def evaluate_second_to_last_policy_sweep(
    paths: Iterable[Path],
    sheet_states,
    *,
    second_to_last_team: int = 1,
    num_angles: int = 10,
    num_speeds: int = 10,
    num_y_vals: int = 6,
) -> pl.DataFrame:
    """Compare NN and grid-search second throws from the same candidate grid.

    Both policies use the same ``ThrowsGridSearcher`` on each sheet state, and
    both use actual-score grid search for the final throw. The result reports
    final points scored by ``second_to_last_team`` and their standard error.
    """
    throw_searcher = bot.ThrowsGridSearcher(num_angles, num_speeds, num_y_vals)
    rows: list[dict[str, float | int | str]] = []
    for path in paths:
        path = Path(path)
        network, normalizer = curling_nn.load_v_weights(path)
        comparison = evaluation.compare_second_to_last_policies(
            sheet_states,
            second_to_last_team=second_to_last_team,
            throw_searcher=throw_searcher,
            value_network=network,
            value_normalizer=normalizer,
        )
        points_column = "team_1_score" if second_to_last_team == 1 else "team_0_score"
        num_parameters = sum(
            layer.weights.size + layer.bias.size
            for layer in network.linear_layers()
        )
        for policy in ("value_network", "grid_search"):
            points = comparison.filter(
                comparison["second_to_last_policy"] == policy
            )[points_column].to_numpy()
            rows.append(
                {
                    "path": str(path),
                    "training_size": int(path.stem.split("_n")[1].split("_h")[0]),
                    "hidden_layer_size": network.hidden_layer_size,
                    "num_parameters": num_parameters,
                    "second_to_last_policy": policy,
                    "average_points": float(np.mean(points)),
                    "points_stderr": float(
                        np.std(points, ddof=1) / np.sqrt(points.size)
                    ),
                }
            )
    return pl.DataFrame(rows).sort(
        ["training_size", "hidden_layer_size", "second_to_last_policy"]
    )
