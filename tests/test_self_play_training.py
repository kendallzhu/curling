import numpy as np

import self_play_training as sp
import sequential_training as seq

CONFIG = sp.SelfPlayConfig(
    start_turn=8,
    max_stones=10,
    num_stones_per_side=5,
    num_states=2,
    grid_size=(2, 2, 2),
    hidden_layer_size=4,
)
BOARDS_PER_GAME = CONFIG.max_stones - CONFIG.start_turn
ROWS_PER_ITERATION = BOARDS_PER_GAME * CONFIG.num_states


def _untrained_model():
    model = seq.make_model(
        seed=0,
        max_stones=CONFIG.max_stones,
        num_stones_per_side=CONFIG.num_stones_per_side,
        hidden_layer_size=CONFIG.hidden_layer_size,
    )
    width = seq.feature_width(CONFIG.max_stones)
    return model, seq.dataset.Normalizer(np.zeros(width), np.ones(width))


def test_generate_self_play_batch_should_create_one_row_per_pre_final_throw_board():
    model, normalizer = _untrained_model()

    data = sp.generate_self_play_batch(
        model, normalizer, config=CONFIG, rng=np.random.default_rng(0)
    )

    assert data.sheet_states.x.shape == (ROWS_PER_ITERATION, CONFIG.max_stones)
    assert data.final_scores.shape == (ROWS_PER_ITERATION,)
    assert np.all(np.abs(data.final_scores) <= CONFIG.num_stones_per_side)
