"""Generate the shared starting states for the second-to-last policy demo."""

from pathlib import Path

import evaluation


if __name__ == "__main__":
    path = Path("scratch/second_to_last_evaluation_states.npz")
    states = evaluation.generate_second_to_last_evaluation_states()
    evaluation.write_sheet_states(path, states)
    print(f"wrote {states.x.shape[0]} starting states to {path}")
