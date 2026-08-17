import numpy as np

import bot
import curling_nn
import dataset
import evaluation


def test_sheet_state_evaluation_round_trip(tmp_path):
    original = evaluation.generate_second_to_last_evaluation_states(
        seed=0, num_sims=2
    )
    path = tmp_path / "states.npz"
    evaluation.write_sheet_states(path, original)
    loaded = evaluation.load_sheet_states(path)

    np.testing.assert_array_equal(loaded.first_team, original.first_team)
    np.testing.assert_allclose(loaded.x, original.x)
    np.testing.assert_allclose(loaded.y, original.y)
    np.testing.assert_allclose(loaded.velocities.v, original.velocities.v)
    np.testing.assert_allclose(loaded.velocities.theta, original.velocities.theta)
    np.testing.assert_array_equal(
        loaded.rotation_directions, original.rotation_directions
    )


def test_second_to_last_policy_comparison_has_both_policies(tmp_path):
    states = evaluation.generate_second_to_last_evaluation_states(
        seed=0, num_sims=2
    )
    value_network = curling_nn.ValueNetwork(
        seed=0, num_stones=9, num_stones_per_side=5
    )
    normalizer = dataset.Normalizer(
        feature_means=np.zeros(46), feature_stdevs=np.ones(46)
    )

    comparison = evaluation.compare_second_to_last_policies(
        states,
        second_to_last_team=0,
        throw_searcher=bot.ThrowsGridSearcher(2, 2, 2),
        value_network=value_network,
        value_normalizer=normalizer,
    )

    assert comparison.columns == [
        "sim_idx",
        "second_to_last_policy",
        "team_0_score",
        "team_1_score",
        "team_0_net_score",
    ]
    assert comparison.height == 4
    assert set(comparison["second_to_last_policy"].to_list()) == {
        "value_network",
        "grid_search",
    }
