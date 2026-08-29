from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import nn
import dataset
import state
from constants import house_outer_circle_radius, STONE_RADIUS_M


def raw_sheet_state_features(sheet_states: state.SheetStates) -> np.ndarray:
    is_thrown = np.where(sheet_states.x == 0, 0, 1)
    distance_from_center = sheet_states.distance_from_center_of_house()
    is_in_house = np.where(
        distance_from_center < house_outer_circle_radius + STONE_RADIUS_M,
        1,
        0,
    )
    num_sims = sheet_states.x.shape[0]
    return np.concatenate(
        [
            sheet_states.first_team.reshape((num_sims, 1)),
            is_thrown,
            sheet_states.x,
            sheet_states.y,
            distance_from_center,
            is_in_house,
        ],
        axis=1,
    )


def _score_match_dataset(
    raw_features: np.ndarray,
    final_scores: np.ndarray,
    num_stones_per_side: int,
) -> dataset.TrainingData:
    score_matches = (
        final_scores.reshape((raw_features.shape[0], 1))
        == np.arange(
            -num_stones_per_side, num_stones_per_side + 1, dtype=int
        ).reshape((1, 2 * num_stones_per_side + 1))
    ).astype(int)
    normalizer = dataset.Normalizer.from_features(raw_features)
    return dataset.TrainingData(
        input_features=normalizer.normalize(raw_features),
        answers=score_matches,
        normalizer=normalizer,
        raw_inputs=raw_features,
    )


class QInputFeatures:
    @staticmethod
    def raw_of_sheet_states(
        sheet_states: state.SheetStates, throws: state.Throws
    ) -> np.ndarray:
        next_team_to_play = sheet_states.next_team_to_play()
        assert (throws.team == next_team_to_play).all()
        num_sims = sheet_states.x.shape[0]
        return np.concatenate(
            [
                raw_sheet_state_features(sheet_states),
                throws.angle_deg.reshape((num_sims, 1)),
                throws.speed.reshape((num_sims, 1)),
                throws.y_val.reshape((num_sims, 1)),
                np.where(throws.turn == 1, 1, 0).reshape((num_sims, 1)),
                np.where(throws.turn == 0, 1, 0).reshape((num_sims, 1)),
                np.where(throws.turn == -1, 1, 0).reshape((num_sims, 1)),
            ],
            axis=1,
        )

    @staticmethod
    def create_of_sheet_states(
        sheet_states: state.SheetStates,
        throws: state.Throws,
        normalizer: dataset.Normalizer,
    ) -> np.ndarray:
        return normalizer.normalize(
            QInputFeatures.raw_of_sheet_states(sheet_states, throws)
        )

    @staticmethod
    def create_score_match_dataset_from_sheet_states(
        sheet_states: state.SheetStates,
        throws: state.Throws,
        final_scores: np.ndarray,
        num_stones_per_side: int,
    ) -> dataset.TrainingData:
        return _score_match_dataset(
            QInputFeatures.raw_of_sheet_states(sheet_states, throws),
            final_scores,
            num_stones_per_side,
        )


class VInputFeatures:
    @staticmethod
    def raw_of_sheet_states(sheet_states: state.SheetStates) -> np.ndarray:
        return raw_sheet_state_features(sheet_states)

    @staticmethod
    def create_score_match_dataset_from_sheet_states(
        sheet_states: state.SheetStates,
        final_scores: np.ndarray,
        num_stones_per_side: int,
    ) -> dataset.TrainingData:
        return _score_match_dataset(
            VInputFeatures.raw_of_sheet_states(sheet_states),
            final_scores,
            num_stones_per_side,
        )

    @staticmethod
    def create_of_sheet_states(
        sheet_states: state.SheetStates,
        normalizer: dataset.Normalizer,
    ) -> np.ndarray:
        return normalizer.normalize(VInputFeatures.raw_of_sheet_states(sheet_states))


def _score_mlp_layers(
    rng: np.random.Generator,
    input_layer_size: int,
    hidden_layer_size: int,
    output_layer_size: int,
) -> list:
    l1 = nn.LinearBatched(
        rng.normal(size=(hidden_layer_size, input_layer_size))
        * np.sqrt(2 / input_layer_size)
    )
    l2 = nn.LinearBatched(
        rng.normal(size=(hidden_layer_size, hidden_layer_size))
        * np.sqrt(2 / hidden_layer_size)
    )
    l3 = nn.LinearBatched(
        rng.normal(size=(hidden_layer_size, hidden_layer_size))
        * np.sqrt(2 / hidden_layer_size)
    )
    l4 = nn.LinearBatched(
        1
        / np.sqrt(hidden_layer_size)
        * rng.normal(size=(output_layer_size, hidden_layer_size))
    )
    return [l1, nn.Max0(), l2, nn.Max0(), l3, nn.Max0(), l4]


class ScoreNetwork(nn.NN):
    def __init__(
        self,
        *,
        seed: int,
        num_stones: int,
        input_layer_size: int,
        hidden_layer_size: int,
        num_stones_per_side: int,
        output_layer_size: int | None = None,
    ):
        self.num_stones = num_stones
        self.hidden_layer_size = hidden_layer_size
        self.num_stones_per_side = num_stones_per_side
        if output_layer_size is None:
            output_layer_size = 2 * num_stones_per_side + 1
        rng = np.random.default_rng(seed)
        super().__init__(
            _score_mlp_layers(
                rng, input_layer_size, hidden_layer_size, output_layer_size
            )
        )

    def linear_layers(self) -> list[nn.LinearBatched]:
        return [layer for layer in self.layers if isinstance(layer, nn.LinearBatched)]

    def expected_score(self, nn_output: np.ndarray) -> np.ndarray:
        """Expected net score from categorical output nodes over [-n, n]."""
        nn_probs = nn.softmax(nn_output)
        weights = nn_probs.reshape(nn_output.shape[0], -1)
        score_values = np.arange(
            -self.num_stones_per_side, self.num_stones_per_side + 1
        )
        assert weights.shape[1] == score_values.shape[0]
        return weights @ score_values


def print_trainable_parameter_info(neural_network: ScoreNetwork) -> int:
    """Print trainable parameter shapes and return their total count."""
    total = 0
    print(
        f"{type(neural_network).__name__}: "
        f"{neural_network.num_stones} input stones, "
        f"hidden layer size {neural_network.hidden_layer_size}"
    )
    for index, layer in enumerate(neural_network.linear_layers()):
        weight_count = layer.weights.size
        bias_count = layer.bias.size
        layer_count = weight_count + bias_count
        total += layer_count
        print(
            f"  linear layer {index}: "
            f"weights {layer.weights.shape}, bias {layer.bias.shape}, "
            f"parameters {layer_count}"
        )
    print(f"total trainable parameters: {total}")
    return total


class QNetwork(ScoreNetwork):
    def __init__(
        self,
        seed: int,
        num_stones: int,
        hidden_layer_size: int = 20,
        output_layer_size: int | None = None,
    ):
        super().__init__(
            seed=seed,
            num_stones=num_stones,
            input_layer_size=5 * num_stones + 7,
            hidden_layer_size=hidden_layer_size,
            num_stones_per_side=(num_stones + 1) // 2,
            output_layer_size=output_layer_size,
        )


class ValueNetwork(ScoreNetwork):
    def __init__(
        self,
        seed: int,
        num_stones: int,
        hidden_layer_size: int = 20,
        num_stones_per_side: int | None = None,
        output_layer_size: int | None = None,
    ):
        if num_stones_per_side is None:
            num_stones_per_side = (num_stones + 1) // 2
        super().__init__(
            seed=seed,
            num_stones=num_stones,
            input_layer_size=5 * num_stones + 1,
            hidden_layer_size=hidden_layer_size,
            num_stones_per_side=num_stones_per_side,
            output_layer_size=output_layer_size,
        )


def _write_score_weights(
    path: str | Path,
    neural_network: ScoreNetwork,
    normalizer: dataset.Normalizer,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "num_stones": np.asarray(neural_network.num_stones),
        "hidden_layer_size": np.asarray(neural_network.hidden_layer_size),
        "num_stones_per_side": np.asarray(neural_network.num_stones_per_side),
        "feature_means": np.asarray(normalizer.feature_means),
        "feature_stdevs": np.asarray(normalizer.feature_stdevs),
    }
    for i, layer in enumerate(neural_network.linear_layers()):
        arrays[f"w{i}"] = layer.weights
        arrays[f"b{i}"] = layer.bias
    np.savez(path, **arrays)


def _load_normalizer_and_layers(
    data, neural_network: ScoreNetwork
) -> dataset.Normalizer:
    for i, layer in enumerate(neural_network.linear_layers()):
        layer.weights = np.array(data[f"w{i}"], copy=True)
        layer.bias = np.array(data[f"b{i}"], copy=True)
    return dataset.Normalizer(
        feature_means=np.array(data["feature_means"], copy=True),
        feature_stdevs=np.array(data["feature_stdevs"], copy=True),
    )


def write_q_weights(
    path: str | Path,
    neural_network: QNetwork,
    normalizer: dataset.Normalizer,
) -> None:
    _write_score_weights(path, neural_network, normalizer)


def load_q_weights(path: str | Path) -> tuple[QNetwork, dataset.Normalizer]:
    with np.load(path) as data:
        neural_network = QNetwork(
            seed=0,
            num_stones=int(data["num_stones"]),
            hidden_layer_size=int(data["hidden_layer_size"]),
        )
        normalizer = _load_normalizer_and_layers(data, neural_network)
    return neural_network, normalizer


def write_v_weights(
    path: str | Path,
    neural_network: ValueNetwork,
    normalizer: dataset.Normalizer,
) -> None:
    _write_score_weights(path, neural_network, normalizer)


def load_v_weights(path: str | Path) -> tuple[ValueNetwork, dataset.Normalizer]:
    with np.load(path) as data:
        num_stones = int(data["num_stones"])
        if "num_stones_per_side" in data:
            num_stones_per_side = int(data["num_stones_per_side"])
        else:
            num_stones_per_side = (num_stones + 1) // 2
        neural_network = ValueNetwork(
            seed=0,
            num_stones=num_stones,
            hidden_layer_size=int(data["hidden_layer_size"]),
            num_stones_per_side=num_stones_per_side,
        )
        normalizer = _load_normalizer_and_layers(data, neural_network)
    return neural_network, normalizer
