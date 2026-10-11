import numpy as np

import bot


def test_perturb_throws_should_only_change_angle_by_allowed_offsets():
    n = 1000
    throws = bot.Throws(
        angle_deg=np.full(n, 10.0),
        speed=np.ones(n),
        turn=np.zeros(n, dtype=int),
        y_val=np.zeros(n),
        team=np.zeros(n, dtype=int),
    )
    out = bot.perturb_throws(throws, rng=np.random.default_rng(0))
    assert set(np.round(out.angle_deg - 10.0, 6)) == {0.0, -0.2, 0.2}
    assert np.array_equal(out.speed, throws.speed)
    assert np.array_equal(out.turn, throws.turn)
