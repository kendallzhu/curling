"""Geometry statistics for sampled throw/score distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants import (
    max_release_angle,
    max_release_speed,
    max_release_y,
    min_release_angle,
    min_release_speed,
    min_release_y,
)
from state import Throws


@dataclass(frozen=True)
class ThrowGeometryStatistics:
    """Neighborhood statistics for every throw attaining the top score."""

    top_score: float
    lower_score: float
    top_indices: np.ndarray
    nearest_lower_indices: np.ndarray
    distances: np.ndarray
    distance_percentiles: np.ndarray


@dataclass(frozen=True)
class NearestThrowScoreDistribution:
    """Average score distribution among each throw's nearest neighbors."""

    score_values: np.ndarray
    probabilities: np.ndarray


def _normalized_coordinates(throws: Throws) -> np.ndarray:
    ranges = np.array(
        [
            [min_release_angle, max_release_angle],
            [min_release_speed, max_release_speed],
            [min_release_y, max_release_y],
        ],
        dtype=float,
    )
    values = np.column_stack((throws.angle_deg, throws.speed, throws.y_val)).astype(float)
    if not np.all((values >= ranges[:, 0]) & (values <= ranges[:, 1])):
        raise ValueError("throw coordinates must lie within the release search bounds")
    return (values - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])


def compute_throw_geometry_statistics(
    throws: Throws,
    scores: np.ndarray,
) -> ThrowGeometryStatistics:
    """Compute top-to-lower-score distances and percentile ranks.

    ``throws`` and ``scores`` describe one sheet state.  For every sampled
    throw with score ``max(scores)``, the nearest throw with score
    ``max(scores) - 1`` and the same ``turn`` is selected.  Distances use
    normalized angle, speed, and y coordinates and Euclidean distance.

    Each top-score distance is ranked against nearest same-spin lower-score
    distances for all sampled throws. A lower-score throw does not match to
    itself, so its own zero-distance entry is excluded.
    """
    scores = np.asarray(scores)
    n_throws = throws.angle_deg.size
    if scores.ndim != 1 or scores.size != n_throws:
        raise ValueError("scores must be one-dimensional with one entry per throw")
    if n_throws == 0:
        raise ValueError("at least one throw is required")
    coordinates = _normalized_coordinates(throws)
    top_score = np.max(scores)
    lower_score = top_score - 1
    top_indices = np.flatnonzero(scores == top_score)
    nearest_indices = np.full(top_indices.size, -1, dtype=int)
    distances = np.full(top_indices.size, np.nan, dtype=float)
    distance_percentiles = np.full(top_indices.size, np.nan, dtype=float)

    for result_index, top_index in enumerate(top_indices):
        candidates = np.flatnonzero(
            (scores == lower_score) & (throws.turn == throws.turn[top_index])
        )
        if candidates.size == 0:
            continue
        candidate_distances = np.linalg.norm(
            coordinates[candidates] - coordinates[top_index], axis=1
        )
        nearest_position = int(np.argmin(candidate_distances))
        nearest_indices[result_index] = candidates[nearest_position]
        distances[result_index] = candidate_distances[nearest_position]

        same_spin = np.flatnonzero(throws.turn == throws.turn[top_index])
        all_distances = np.linalg.norm(
            coordinates[same_spin, None, :] - coordinates[candidates][None, :, :],
            axis=2,
        )
        self_matches = np.flatnonzero(scores[same_spin] == lower_score)
        if self_matches.size:
            candidate_positions = np.searchsorted(candidates, same_spin[self_matches])
            all_distances[self_matches, candidate_positions] = np.inf
        nearest_distances = np.min(all_distances, axis=1)
        nearest_distances = nearest_distances[np.isfinite(nearest_distances)]
        distance_percentiles[result_index] = 100.0 * np.mean(
            nearest_distances <= distances[result_index]
        )

    return ThrowGeometryStatistics(
        top_score=float(top_score),
        lower_score=float(lower_score),
        top_indices=top_indices,
        nearest_lower_indices=nearest_indices,
        distances=distances,
        distance_percentiles=distance_percentiles,
    )


def compute_nearest_throw_score_distribution(
    throws: Throws,
    scores: np.ndarray,
    num_neighbors: int = 10,
    score_values: np.ndarray | None = None,
) -> NearestThrowScoreDistribution:
    """Compute average neighbor-score probabilities for each source score.

    For every throw, the ``num_neighbors`` nearest *other* throws are found in
    normalized angle, speed, and release-y coordinate space.  Each row of the
    returned matrix averages the fraction of those neighbors having each
    possible score, over throws with the row's source score.

    A row is ``nan`` when no throw has the corresponding source score.  If
    ``score_values`` is omitted, the sorted unique scores are used.
    """
    scores = np.asarray(scores)
    n_throws = throws.angle_deg.size
    if scores.ndim != 1 or scores.size != n_throws:
        raise ValueError("scores must be one-dimensional with one entry per throw")
    if n_throws < 2:
        raise ValueError("at least two throws are required")
    if not 1 <= num_neighbors < n_throws:
        raise ValueError("num_neighbors must be between 1 and the number of other throws")

    coordinates = _normalized_coordinates(throws)
    if score_values is None:
        score_values = np.unique(scores)
    else:
        score_values = np.asarray(score_values)
    if score_values.ndim != 1 or score_values.size == 0:
        raise ValueError("score_values must be a non-empty one-dimensional array")
    if not np.all(np.isin(scores, score_values)):
        raise ValueError("score_values must include every score in scores")

    neighbor_scores = np.empty((n_throws, num_neighbors), dtype=scores.dtype)
    for throw_index in range(n_throws):
        distances = np.linalg.norm(coordinates - coordinates[throw_index], axis=1)
        distances[throw_index] = np.inf
        nearest_indices = np.argsort(distances)[:num_neighbors]
        neighbor_scores[throw_index] = scores[nearest_indices]

    probabilities = np.full((score_values.size, score_values.size), np.nan, dtype=float)
    for source_index, source_score in enumerate(score_values):
        source_neighbors = neighbor_scores[scores == source_score]
        for target_index, target_score in enumerate(score_values):
            probabilities[source_index, target_index] = np.mean(
                source_neighbors == target_score
            )

    return NearestThrowScoreDistribution(
        score_values=score_values,
        probabilities=probabilities,
    )
