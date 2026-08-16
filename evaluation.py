"""Generation and preparation of fixed evaluation datasets."""

from __future__ import annotations

import numpy as np

import data_generation
import dataset


def apply_model_normalizer(
    raw_data: dataset.TrainingData,
    normalizer: dataset.Normalizer,
) -> dataset.TrainingData:
    """Apply a model's saved normalizer to raw evaluation features."""
    if raw_data.raw_inputs is None:
        raise ValueError("raw_data.raw_inputs is required")
    return dataset.TrainingData(
        input_features=normalizer.normalize(raw_data.raw_inputs),
        answers=raw_data.answers,
        normalizer=normalizer,
        raw_inputs=raw_data.raw_inputs,
    )


def generate_q_evaluation_data(
    normalizer: dataset.Normalizer,
    *,
    seed: int = 2026,
    num_sims: int = 300,
    team: int = 1,
    n_random_throws: int = 1,
    n_per_score: int = 5,
    num_stones_per_side: int = 5,
) -> dataset.TrainingData:
    """Generate Q-network evaluation rows using the standard Q data builder."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    sheet_states = data_generation.random_sheet_states(
        team1=num_stones_per_side,
        team2=num_stones_per_side - 1,
        num_sims=num_sims,
    )
    raw_data = data_generation.q_network_training_data(
        sheet_states=sheet_states,
        team=team,
        rng=rng,
        n_random_throws=n_random_throws,
        n_per_score=n_per_score,
        num_stones_per_side=num_stones_per_side,
    )
    return apply_model_normalizer(raw_data, normalizer)


def generate_value_evaluation_data(
    normalizer: dataset.Normalizer,
    *,
    seed: int = 2026,
    num_sims: int = 100,
    team: int = 1,
    num_stones_per_side: int = 5,
) -> dataset.TrainingData:
    """Generate value-network evaluation rows using grid-searched final throws."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    sheet_states = data_generation.random_sheet_states(
        team1=num_stones_per_side,
        team2=num_stones_per_side - 2,
        num_sims=num_sims,
    )
    raw_data = data_generation.value_network_training_data(
        sheet_states=sheet_states,
        team=team,
        rng=rng,
        num_stones_per_side=num_stones_per_side,
    )
    return apply_model_normalizer(raw_data, normalizer)


def generate_evaluation_data(
    q_normalizer: dataset.Normalizer,
    value_normalizer: dataset.Normalizer,
    *,
    seed: int = 2026,
    q_num_sims: int = 300,
    value_num_sims: int = 100,
    team: int = 1,
    num_stones_per_side: int = 5,
    n_random_throws: int = 1,
    n_per_score: int = 5,
) -> tuple[dataset.TrainingData, dataset.TrainingData]:
    """Generate ``(q_network_data, value_network_data)`` for evaluation."""
    q_data = generate_q_evaluation_data(
        q_normalizer,
        seed=seed,
        num_sims=q_num_sims,
        team=team,
        n_random_throws=n_random_throws,
        n_per_score=n_per_score,
        num_stones_per_side=num_stones_per_side,
    )
    value_data = generate_value_evaluation_data(
        value_normalizer,
        seed=seed,
        num_sims=value_num_sims,
        team=team,
        num_stones_per_side=num_stones_per_side,
    )
    return q_data, value_data
