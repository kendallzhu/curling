from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import nn
import dataset
import state
from constants import house_outer_circle_radius, STONE_RADIUS_M


VALUE_FEATURE_FORMAT_VERSION = 1


def value_feature_width(max_stones: int) -> int:
    """Return the feature width for the padded self-play value format."""
    return 1 + 6 * max_stones


def validate_value_checkpoint_format(data) -> int:
    """Validate a value checkpoint and return its padded stone-slot count."""
    num_stones = int(data["num_stones"])
    if "value_feature_format_version" in data:
        version = int(data["value_feature_format_version"])
        if version != VALUE_FEATURE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported value feature format version {version}; "
                f"expected {VALUE_FEATURE_FORMAT_VERSION}"
            )
    input_width = int(data["w0"].shape[1])
    expected_input_width = value_feature_width(num_stones)
    if input_width != expected_input_width:
        raise ValueError(
            f"incompatible value checkpoint: expected {expected_input_width} "
            f"padded value features, found {input_width}"
        )
    return num_stones


def raw_padded_sheet_state_features(
    sheet_states: state.SheetStates, max_stones: int
) -> np.ndarray:
    """Encode variable-length boards in the fixed-width self-play format."""
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
            distance[:, :n_stones] < house_outer_circle_radius + STONE_RADIUS_M
        )
    thrown = np.zeros_like(x)
    thrown[:, :n_stones] = 1.0
    not_thrown = 1.0 - thrown
    return np.concatenate(
        [sheet_states.first_team[:, None], not_thrown, thrown, x, y, distance, in_house],
        axis=1,
    )


def _score_match_dataset(
    raw_features: np.ndarray,
    final_scores: np.ndarray,
    num_stones_per_side: int,
    mask: np.ndarray | None = None,
) -> dataset.TrainingData:
    score_matches = (
        final_scores.reshape((raw_features.shape[0], 1))
        == np.arange(
            -num_stones_per_side, num_stones_per_side + 1, dtype=int
        ).reshape((1, 2 * num_stones_per_side + 1))
    ).astype(int)
    normalizer = dataset.Normalizer.from_features(raw_features, mask)
    return dataset.TrainingData(
        input_features=normalizer.normalize(raw_features, mask),
        answers=score_matches,
        normalizer=normalizer,
        raw_inputs=raw_features,
        mask=mask,
    )


class ValueInputFeatures:
    @staticmethod
    def raw_of_sheet_states(
        sheet_states: state.SheetStates, max_stones: int
    ) -> np.ndarray:
        return raw_padded_sheet_state_features(sheet_states, max_stones)

    @staticmethod
    def create_score_match_dataset_from_sheet_states(
        sheet_states: state.SheetStates,
        final_scores: np.ndarray,
        num_stones_per_side: int,
        max_stones: int,
    ) -> dataset.TrainingData:
        return _score_match_dataset(
            ValueInputFeatures.raw_of_sheet_states(sheet_states, max_stones),
            final_scores,
            num_stones_per_side,
        )

    @staticmethod
    def create_of_sheet_states(
        sheet_states: state.SheetStates,
        normalizer: dataset.Normalizer,
        max_stones: int,
    ) -> np.ndarray:
        return normalizer.normalize(
            ValueInputFeatures.raw_of_sheet_states(sheet_states, max_stones)
        )


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
            input_layer_size=value_feature_width(num_stones),
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
        "value_feature_format_version": np.asarray(VALUE_FEATURE_FORMAT_VERSION),
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


def write_v_weights(
    path: str | Path,
    neural_network: ValueNetwork,
    normalizer: dataset.Normalizer,
) -> None:
    _write_score_weights(path, neural_network, normalizer)


def load_v_weights(
    path: str | Path,
) -> tuple[ValueNetwork, dataset.Normalizer]:
    with np.load(path) as data:
        num_stones = validate_value_checkpoint_format(data)
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
