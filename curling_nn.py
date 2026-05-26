import numpy as np

import nn
import dataset
import state
import scoring
from constants import house_outer_circle_radius, STONE_RADIUS_M


class InputFeatures:
    @staticmethod
    def create_of_sheet_states(sheet_states: state.SheetStates) -> np.ndarray:
        is_thrown = np.where(sheet_states.x == 0, 0, 1)
        distance_from_center = sheet_states.distance_from_center_of_house()
        is_in_house = np.where(
            distance_from_center < house_outer_circle_radius + STONE_RADIUS_M,
            1,
            0,
        )
        raw_features = np.concatenate(
            [
                is_thrown,
                sheet_states.team,
                sheet_states.x,
                sheet_states.y,
                distance_from_center,
                is_in_house,
            ],
            axis=1,
        )
        normalizer = dataset.Normalizer.from_features(raw_features)
        return normalizer.normalize(raw_features)

    @staticmethod
    def create_score_match_dataset_from_sheet_states(
        sheet_states: state.SheetStates,
        num_stones_per_side: int,
    ) -> dataset.TrainingData:
        score = scoring.get_net_score_for_team(sheet_states, 0)
        score_matches = (
            score.reshape((sheet_states.x.shape[0], 1))
            == np.arange(
                -num_stones_per_side, num_stones_per_side + 1, dtype=int
            ).reshape((1, 2 * num_stones_per_side + 1))
        ).astype(int)
        raw_features = InputFeatures.create_of_sheet_states(sheet_states)
        normalizer = dataset.Normalizer.from_features(raw_features)
        return dataset.TrainingData(
            input_features=normalizer.normalize(raw_features),
            answers=score_matches,
            normalizer=normalizer,
            raw_inputs=raw_features,
        )


class ValueNetwork(nn.NN):
    def __init__(
        self,
        seed: int,
        num_stones_per_side: int,
        hidden_layer_size: int = 20,
        output_layer_size: int = 1
    ):
        rng = np.random.default_rng(seed)

        input_layer_size = 6 * 2 * num_stones_per_side

        l1 = nn.LinearBatched(
            rng.normal(size=(hidden_layer_size, input_layer_size))
            * np.sqrt(2 / input_layer_size)
        )
        act1 = nn.Max0()

        l2 = nn.LinearBatched(
            rng.normal(size=(hidden_layer_size, hidden_layer_size))
            * np.sqrt(2 / hidden_layer_size)
        )
        act2 = nn.Max0()

        l3 = nn.LinearBatched(
            rng.normal(size=(hidden_layer_size, hidden_layer_size))
            * np.sqrt(2 / hidden_layer_size)
        )
        act3 = nn.Max0()

        l4 = nn.LinearBatched(
            1 / np.sqrt(hidden_layer_size) * rng.normal(size=(output_layer_size, hidden_layer_size))
        )

        layers = [l1, act1, l2, act2, l3, act3, l4]
        layers.append(nn.MapTo01())
        super().__init__(layers)
