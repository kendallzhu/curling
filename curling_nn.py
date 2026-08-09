import numpy as np

import nn
import dataset
import state
from constants import house_outer_circle_radius, STONE_RADIUS_M


class InputFeatures:
    @staticmethod
    def raw_of_sheet_states(
        sheet_states: state.SheetStates, throws: state.Throws
    ) -> np.ndarray:
        is_thrown = np.where(sheet_states.x == 0, 0, 1)
        distance_from_center = sheet_states.distance_from_center_of_house()
        is_in_house = np.where(
            distance_from_center < house_outer_circle_radius + STONE_RADIUS_M,
            1,
            0,
        )
        next_team_to_play = sheet_states.next_team_to_play()
        assert (throws.team == next_team_to_play).all()
        num_sims = sheet_states.x.shape[0]
        return np.concatenate(
            [
                sheet_states.first_team.reshape((num_sims, 1)),
                is_thrown,
                sheet_states.x,
                sheet_states.y,
                distance_from_center,
                is_in_house,
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
            InputFeatures.raw_of_sheet_states(sheet_states, throws)
        )

    @staticmethod
    def create_score_match_dataset_from_sheet_states(
        sheet_states: state.SheetStates,
        throws: state.Throws,
        final_scores: np.ndarray,
        num_stones_per_side: int,
    ) -> dataset.TrainingData:
        score_matches = (
            final_scores.reshape((sheet_states.x.shape[0], 1))
            == np.arange(
                -num_stones_per_side, num_stones_per_side + 1, dtype=int
            ).reshape((1, 2 * num_stones_per_side + 1))
        ).astype(int)
        raw_features = InputFeatures.raw_of_sheet_states(sheet_states, throws)
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
        num_stones: int,
        hidden_layer_size: int = 20,
        output_layer_size: int | None = None,
    ):
        self.num_stones_per_side = ((num_stones + 1) // 2)
        if output_layer_size is None:
            output_layer_size = 2 * self.num_stones_per_side + 1
        rng = np.random.default_rng(seed)

        # TODO: clarify
        input_layer_size = 5 * num_stones + 7

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
            1
            / np.sqrt(hidden_layer_size)
            * rng.normal(size=(output_layer_size, hidden_layer_size))
        )

        layers = [l1, act1, l2, act2, l3, act3, l4]

        super().__init__(layers)


    def expected_score(self, nn_output: np.ndarray) -> np.ndarray:
        """Expected net score from categorical output nodes over [-n, n]."""
        nn_probs = nn.softmax(nn_output)
        weights = nn_probs.reshape(nn_output.shape[0], -1)
        score_values = np.arange(
            -self.num_stones_per_side, self.num_stones_per_side + 1
        )
        assert weights.shape[1] == score_values.shape[0]
        return weights @ score_values
