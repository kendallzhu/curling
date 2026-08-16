from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    def size(self) -> int:
        return int(self.input_features.shape[0])

    def __len__(self) -> int:
        return self.size()

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



    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:
        return np.where(
            np.std(X, axis=0) == 0,
            np.zeros(X.shape),
            (X - np.mean(X, axis=0)) / np.std(X, axis=0),
        )

    def partition(
        self,
        validation_fraction: float = 0.2,
        *,
        seed: int | None = None,
    ) -> tuple["TrainingData", "TrainingData"]:
        """Split into (train, validation), sharing one normalizer fit on train.

        Prefer this over generating a separate validation dataset so both splits
        use the same feature normalization.
        """
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {validation_fraction}"
            )
        n = self.input_features.shape[0]
        if n < 2:
            raise ValueError("Need at least 2 samples to partition")
        n_val = int(round(n * validation_fraction))
        n_val = max(1, min(n - 1, n_val))

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        if self.raw_inputs is not None:
            train_raw = self.raw_inputs[train_idx]
            val_raw = self.raw_inputs[val_idx]
            normalizer = Normalizer.from_features(train_raw)
            train = TrainingData(
                input_features=normalizer.normalize(train_raw),
                answers=self.answers[train_idx],
                normalizer=normalizer,
                raw_inputs=train_raw,
            )
            validation = TrainingData(
                input_features=normalizer.normalize(val_raw),
                answers=self.answers[val_idx],
                normalizer=normalizer,
                raw_inputs=val_raw,
            )
            return train, validation

        return (
            TrainingData(
                input_features=self.input_features[train_idx],
                answers=self.answers[train_idx],
                normalizer=self.normalizer,
                raw_inputs=None,
            ),
            TrainingData(
                input_features=self.input_features[val_idx],
                answers=self.answers[val_idx],
                normalizer=self.normalizer,
                raw_inputs=None,
            ),
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
            _, ax = plt.subplots(figsize=figsize)
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


def write_training_data(path: str | Path, data: TrainingData) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "input_features": data.input_features,
        "answers": data.answers,
        "feature_means": data.normalizer.feature_means,
        "feature_stdevs": data.normalizer.feature_stdevs,
    }
    if data.raw_inputs is not None:
        arrays["raw_inputs"] = data.raw_inputs
    np.savez(path, **arrays)


def load_training_data(path: str | Path) -> TrainingData:
    with np.load(path) as npz:
        raw_inputs = (
            np.array(npz["raw_inputs"], copy=True) if "raw_inputs" in npz.files else None
        )
        return TrainingData(
            input_features=np.array(npz["input_features"], copy=True),
            answers=np.array(npz["answers"], copy=True),
            normalizer=Normalizer(
                feature_means=np.array(npz["feature_means"], copy=True),
                feature_stdevs=np.array(npz["feature_stdevs"], copy=True),
            ),
            raw_inputs=raw_inputs,
        )
