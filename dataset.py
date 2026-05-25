from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt

import state
import presets
import scoring


@dataclass(frozen=True)
class TrainingBatch:
    input_features: np.ndarray
    answers: np.ndarray


@dataclass
class Normalizer:
    feature_means: np.ndarray
    feature_stdevs: np.ndarray

    @classmethod
    def from_features(cls, X):
        return cls(feature_means=np.mean(X, axis=0), feature_stdevs=np.std(X, axis=0))

    def normalize(self, X):
        return np.where(
            self.feature_stdevs == 0,
            X,
            (X - self.feature_means)
            / np.where(self.feature_stdevs == 0, 1, self.feature_stdevs),
        )


@dataclass
class TrainingData:
    input_features: np.ndarray
    answers: np.ndarray
    normalizer: Normalizer
    raw_inputs: np.ndarray | None = None

    @classmethod
    def spiral(
        cls,
        num_points: int = 10000,
        seed: int = 42,
        radius_scale: float = 6 * np.pi,
    ) -> "TrainingData":
        rng = np.random.default_rng(seed)
        theta = radius_scale * rng.random(num_points).reshape((num_points, 1))
        r = radius_scale * rng.random(num_points).reshape((num_points, 1))
        answers = (np.abs((r - theta) % (2 * np.pi)).flatten() < np.pi).astype(
            np.float64
        )
        raw_inputs = r * np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
        normalizer = Normalizer.from_features(raw_inputs)
        input_features = normalizer.normalize(raw_inputs)
        return cls(
            input_features=input_features,
            answers=answers,
            normalizer=normalizer,
            raw_inputs=raw_inputs,
        )

    @classmethod
    def curling(
        cls,
        sheet_states: list[state.SheetStates],
        throws_remaining_by_team: list[tuple[int, int]],
        final_scores: list[np.ndarray],
        seed: int,
        normalizer: Normalizer | None,
    ) -> "TrainingData":
        rng = np.random.default_rng(seed)
        raw_inputs = np.concatenate(
            [
                ss.shuffle_stones(rng).to_input_features(throws_remaining_by_team=tr)
                for (ss, tr) in zip(sheet_states, throws_remaining_by_team)
            ],
            axis=0,
        )
        normalizer = normalizer or Normalizer.from_features(raw_inputs)
        input_features = normalizer.normalize(raw_inputs)
        return cls(
            input_features=input_features,
            answers=np.concatenate(final_scores).reshape((raw_inputs.shape[0], 1)),
            normalizer=normalizer,
            raw_inputs=raw_inputs,
        )

    @classmethod
    def curling_random_sheet_states(
        cls,
        num_sims: int = 10000,
        num_stones_per_side: int = 5,
        throws_remaining_by_team: tuple[int, int] = (0, 0),
    ) -> "TrainingData":
        states = state.concat(
            [
                presets.random_sheet_states(
                    team1=num_stones_per_side, team2=num_stones_per_side
                )
                for _ in range(num_sims)
            ]
        )

        raw_inputs = states.to_input_features(
            throws_remaining_by_team=throws_remaining_by_team
        )
        score = scoring.get_score(states) @ np.array([1, -1])
        score_matches = (
            score.reshape((num_sims, 1))
            == np.arange(
                -num_stones_per_side, num_stones_per_side + 1, dtype=int
            ).reshape((1, 2 * num_stones_per_side + 1))
        ).astype(int)

        answers = np.concatenate([score.reshape((num_sims, 1)), score_matches], axis=1)

        normalizer = Normalizer.from_features(raw_inputs)
        input_features = normalizer.normalize(raw_inputs)
        return cls(
            input_features=input_features,
            answers=score.reshape((num_sims, 1)),
            normalizer=normalizer,
            raw_inputs=raw_inputs,
        )

    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:
        return np.where(
            np.std(X, axis=0) == 0,
            np.zeros(X.shape),
            (X - np.mean(X, axis=0)) / np.std(X, axis=0),
        )

    def shuffle_batches(
        self,
        num_points_per_batch: int,
        seed: int | None = None,
    ) -> list[TrainingBatch]:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.input_features.shape[0])
        batches: list[TrainingBatch] = []
        for batch_min_index in range(0, len(indices), num_points_per_batch):
            batch_indices = indices[
                batch_min_index : batch_min_index + num_points_per_batch
            ]
            batches.append(
                TrainingBatch(
                    input_features=self.input_features[batch_indices],
                    answers=self.answers[batch_indices],
                )
            )
        return batches

    def plot_data(self, ax=None, figsize=(10, 10), size: int = 20):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            self.input_features[self.answers == 0, 0],
            self.input_features[self.answers == 0, 1],
            color="r",
            s=size,
        )
        ax.scatter(
            self.input_features[self.answers == 1, 0],
            self.input_features[self.answers == 1, 1],
            color="g",
            s=size,
        )
        return ax
