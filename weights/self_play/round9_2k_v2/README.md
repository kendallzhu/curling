# Self-play checkpoints: round9_2k_v2

`checkpoints/m_1.npz` through `checkpoints/m_10.npz` are the checkpoints from the 10-iteration `round9_2k_v2` run. Each NPZ contains the value-network weights and feature normalizer. The category-level `schema.json` documents every saved array.

`final_model.npz` is a relative symlink to `checkpoints/m_10.npz`, the final checkpoint.

Load it with the sequential-training helper:

```python
import sequential_training

model, normalizer = sequential_training.load_model(
    "weights/self_play/round9_2k_v2/checkpoints/m_10.npz"
)
```

`stats/training_stats.json` is the compact per-iteration record used when resuming training. `stats/training_metrics.json` feeds the checkpoint plots, and `stats/greedy_comparison.json` contains the paired greedy results. `stats/training_and_evaluation_stats.json` is a legacy combined snapshot from the earlier evaluation workflow; current code does not read or update it.
