"""Generate the fixed datasets used by the calibration stats notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import curling_nn
import dataset
import evaluation
from constants import q_network_weights_path, value_network_weights_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("scratch"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    _, q_normalizer = curling_nn.load_q_weights(q_network_weights_path)
    _, value_normalizer = curling_nn.load_v_weights(value_network_weights_path)
    q_data = evaluation.generate_q_evaluation_data(q_normalizer, seed=args.seed)
    value_data = evaluation.generate_value_evaluation_data(
        value_normalizer, seed=args.seed
    )

    dataset.write_training_data(args.output_dir / "q_calibration_evaluation.npz", q_data)
    dataset.write_training_data(
        args.output_dir / "value_calibration_evaluation.npz", value_data
    )
    print(f"wrote {q_data.size()} Q-network rows")
    print(f"wrote {value_data.size()} value-network rows")


if __name__ == "__main__":
    main()
