import numpy as np

from scratch.throw_geometry_statistics import (
    compute_nearest_throw_score_distribution,
    compute_throw_geometry_statistics,
)
from state import Throws


def test_geometry_statistics_uses_same_spin_and_normalized_coordinates():
    throws = Throws(
        angle_deg=np.array([-4.0, 4.0, -4.0, 0.0]),
        speed=np.array([2.0, 2.0, 2.0, 2.25]),
        turn=np.array([1, 1, -1, 1]),
        y_val=np.array([2.25, 2.25, 2.25, 2.5]),
        team=np.zeros(4, dtype=int),
    )
    result = compute_throw_geometry_statistics(
        throws, np.array([2, 1, 1, 2])
    )

    assert result.top_score == 2
    assert result.lower_score == 1
    np.testing.assert_array_equal(result.top_indices, [0, 3])
    np.testing.assert_array_equal(result.nearest_lower_indices, [1, 1])
    np.testing.assert_allclose(result.distances, [1.0, np.sqrt(0.75)], atol=1e-12)
    assert np.all((result.distance_percentiles >= 0) & (result.distance_percentiles <= 100))


def test_geometry_statistics_returns_nan_when_lower_score_spin_is_missing():
    throws = Throws(
        angle_deg=np.array([0.0, 0.0]),
        speed=np.array([2.25, 2.25]),
        turn=np.array([1, -1]),
        y_val=np.array([2.5, 2.5]),
        team=np.zeros(2, dtype=int),
    )
    result = compute_throw_geometry_statistics(
        throws, np.array([3, 2])
    )

    assert result.nearest_lower_indices[0] == -1
    assert np.isnan(result.distances[0])
    assert np.isnan(result.distance_percentiles[0])


def test_nearest_throw_score_distribution_excludes_the_throw_itself():
    throws = Throws(
        angle_deg=np.array([-4.0, 0.0, 4.0]),
        speed=np.full(3, 2.25),
        turn=np.ones(3, dtype=int),
        y_val=np.full(3, 2.5),
        team=np.zeros(3, dtype=int),
    )

    result = compute_nearest_throw_score_distribution(
        throws, np.array([0, 0, 1]), num_neighbors=2
    )

    np.testing.assert_array_equal(result.score_values, [0, 1])
    np.testing.assert_allclose(result.probabilities, [[0.5, 0.5], [1.0, 0.0]])
