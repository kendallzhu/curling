from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class TrainingBatch:
    input_features: np.ndarray
    answers: np.ndarray


@dataclass
class TrainingData:
    input_features: np.ndarray
    answers: np.ndarray
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
        answers = (np.abs((r - theta) % (2 * np.pi)).flatten() < np.pi).astype(np.float64)
        raw_inputs = r * np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
        input_features = cls.normalize(raw_inputs)
        return cls(input_features=input_features, answers=answers, raw_inputs=raw_inputs)

    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def shuffle_batches(
        self,
        num_points_per_batch: int,
        seed: int | None = None,
    ) -> list[TrainingBatch]:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.input_features.shape[0])
        batches: list[TrainingBatch] = []
        for batch_min_index in range(0, len(indices), num_points_per_batch):
            batch_indices = indices[batch_min_index : batch_min_index + num_points_per_batch]
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
        ax.scatter(self.input_features[self.answers == 0, 0], self.input_features[self.answers == 0, 1], color="r", s=size)
        ax.scatter(self.input_features[self.answers == 1, 0], self.input_features[self.answers == 1, 1], color="g", s=size)
        return ax
