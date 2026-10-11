"""Configuration helpers for ``self_play_training.ipynb``.

Keep the notebook focused on editable experiment settings; this module builds
the derived paths, training config, and run metadata from those settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import self_play_training as self_play
import sequential_training as seq
from bot import ThrowPerturbations


@dataclass(frozen=True)
class NotebookSetup:
    sequential_artifact_dir: Path
    sequential_weights_dir: Path
    initial_model_path: Path
    initial_dataset_path: Path
    self_play_artifact_dir: Path
    self_play_weights_dir: Path
    config: self_play.SelfPlayConfig
    run_parameters: dict


_NOTEBOOK_SETTING_NAMES = (
    "repo_root",
    "N",
    "seed",
    "grid_size",
    "robust_top_fraction",
    "perturbations",
    "rows_per_model",
    "initial_training_steps",
    "random_throw_count",
    "policy_comparison_num_sims",
    "state_generator_version",
    "data_root",
    "sequential_run_name",
    "sequential_run_dir",
    "legacy_sequential_run_dir",
    "sequential_artifact_dir",
    "sequential_weights_dir",
    "initial_model_path",
    "initial_dataset_path",
    "sequential_diagnostic_output_dir",
    "num_iterations",
    "num_states_per_iteration",
    "self_play_start_turn",
    "halflife",
    "hidden_layer_size",
    "batch_size",
    "training_iterations",
    "learning_rate",
    "resume_from",
    "self_play_run_name",
    "self_play_artifact_dir",
    "self_play_weights_dir",
)


def make_notebook_setup(namespace: Mapping[str, Any]) -> NotebookSetup:
    """Build setup from the experiment settings defined in the notebook cell."""
    settings = {name: namespace[name] for name in _NOTEBOOK_SETTING_NAMES}
    return _make_notebook_setup(**settings)


def _make_notebook_setup(
    *,
    repo_root: str | Path,
    N: int,
    seed: int,
    grid_size: tuple[int, int, int],
    robust_top_fraction: float,
    perturbations: ThrowPerturbations,
    rows_per_model: int,
    initial_training_steps: int,
    random_throw_count: int,
    policy_comparison_num_sims: int,
    state_generator_version: str,
    data_root: str | Path,
    sequential_run_name: str | None,
    sequential_run_dir: str | Path | None,
    legacy_sequential_run_dir: str | Path | None,
    sequential_artifact_dir: str | Path | None,
    sequential_weights_dir: str | Path | None,
    initial_model_path: str | Path | None,
    initial_dataset_path: str | Path | None,
    sequential_diagnostic_output_dir: str | Path | None,
    num_iterations: int,
    num_states_per_iteration: int,
    self_play_start_turn: int,
    halflife: float,
    hidden_layer_size: int,
    batch_size: int,
    training_iterations: int,
    learning_rate: float,
    resume_from: int | None,
    self_play_run_name: str,
    self_play_artifact_dir: str | Path | None,
    self_play_weights_dir: str | Path | None,
) -> NotebookSetup:
    """Build paths, self-play settings, and metadata for the notebook run.

    The optional directory and checkpoint arguments let a notebook user
    override any generated artifact path while keeping the usual layout as the
    default.
    """
    repo_root = Path(repo_root).resolve()
    data_root = Path(data_root)

    default_sequential_run_name = (
        f"round{2 * N - 1}_{rows_per_model // 1000}k"
    )
    sequential_run_name = sequential_run_name or default_sequential_run_name
    sequential_run_dir = Path(
        sequential_run_dir
        if sequential_run_dir is not None
        else data_root / f"N{N}_rows{rows_per_model}_round{2 * N - 1}"
    )
    legacy_sequential_run_dir = Path(
        legacy_sequential_run_dir
        if legacy_sequential_run_dir is not None
        else data_root / f"N{N}_rows{rows_per_model}_stages2"
    )
    if sequential_artifact_dir is None:
        if not sequential_run_dir.exists() and legacy_sequential_run_dir.exists():
            sequential_run_dir = legacy_sequential_run_dir
        sequential_artifact_dir = sequential_run_dir / "sequential_experiment"

    sequential_artifact_dir = Path(sequential_artifact_dir)
    sequential_weights_dir = Path(
        sequential_weights_dir
        if sequential_weights_dir is not None
        else seq.training_run_weights_dir("sequential_training", sequential_run_name)
    )
    initial_model_path = Path(
        initial_model_path
        if initial_model_path is not None
        else sequential_weights_dir / "checkpoints" / f"m_{2 * N - 1}.npz"
    )
    initial_dataset_path = Path(
        initial_dataset_path
        if initial_dataset_path is not None
        else sequential_artifact_dir / f"D_{2 * N - 1}.npz"
    )

    config = self_play.SelfPlayConfig(
        num_states=num_states_per_iteration,
        num_iterations=num_iterations,
        grid_size=grid_size,
        start_turn=self_play_start_turn,
        halflife=halflife,
        max_stones=2 * N,
        num_stones_per_side=N,
        hidden_layer_size=hidden_layer_size,
        batch_size=batch_size,
        training_iterations=training_iterations,
        learning_rate=learning_rate,
        robust_top_fraction=robust_top_fraction,
        perturbations=perturbations,
    )

    self_play_artifact_dir = Path(
        self_play_artifact_dir
        if self_play_artifact_dir is not None
        else data_root / f"self_play_N{N}_{self_play_run_name}"
    )
    self_play_weights_dir = Path(
        self_play_weights_dir
        if self_play_weights_dir is not None
        else seq.training_run_weights_dir("self_play", self_play_run_name)
    )

    run_parameters = {
        "N": N,
        "seed": seed,
        "initial_sequential_training": {
            "run_name": sequential_run_name,
            "turn": 2 * N - 1,
            "training_stages": 1,
            "resume_from": None,
            "rows_per_model": rows_per_model,
            "num_training_iterations": initial_training_steps,
            "state_generator_version": state_generator_version,
            "grid_size": list(grid_size),
            "random_throw_count": random_throw_count,
            "robust_top_fraction": robust_top_fraction,
            "perturbations": asdict(perturbations),
            "policy_comparison_num_sims": policy_comparison_num_sims,
            "training_data_generation": {
                "method": (
                    "Sample second-to-last-turn boards, select the final throw, "
                    "and label each board with the resulting team-0 net score."
                ),
                "rows_per_model": rows_per_model,
            },
            "network_training": {
                "hidden_layer_size": 20,
                "batch_size": 100,
                "loss": "softmax cross-entropy",
                "optimizer": "mini-batch gradient descent",
                "regularization": 0.0,
                "learning_rate_schedule": (
                    "0.5 * (1 + cos(pi * iteration / num_training_iterations))"
                ),
                "validation_fraction": 0.2,
                "batch_shuffle_seed": "model index + iteration",
            },
        },
        "self_play": {
            "run_name": self_play_run_name,
            "config": asdict(config),
            "resume_from": resume_from,
        },
        "artifacts": {
            "data_root": data_root.name,
            "initial_model": str(initial_model_path.relative_to(repo_root)),
            "initial_dataset": str(initial_dataset_path.relative_to(data_root)),
            "sequential_artifact_dir": str(
                sequential_artifact_dir.relative_to(data_root)
            ),
            "sequential_weights_dir": str(
                sequential_weights_dir.relative_to(repo_root)
            ),
            "self_play_artifact_dir": str(
                self_play_artifact_dir.relative_to(data_root)
            ),
            "self_play_weights_dir": str(
                self_play_weights_dir.relative_to(repo_root)
            ),
            "diagnostic_output_dir": (
                str(sequential_diagnostic_output_dir)
                if sequential_diagnostic_output_dir
                else None
            ),
        },
    }

    return NotebookSetup(
        sequential_artifact_dir=sequential_artifact_dir,
        sequential_weights_dir=sequential_weights_dir,
        initial_model_path=initial_model_path,
        initial_dataset_path=initial_dataset_path,
        self_play_artifact_dir=self_play_artifact_dir,
        self_play_weights_dir=self_play_weights_dir,
        config=config,
        run_parameters=run_parameters,
    )
