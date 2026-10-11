import user_interface
from state import Throw, empty_board


def test_apply_throw_should_retain_perturbation_offset_and_chance(monkeypatch):
    throw = Throw(angle_deg=0.0, speed=2.0, turn=0, y_val=2.5, team=0)
    released_throw = Throw(angle_deg=0.2, speed=2.0, turn=0, y_val=2.5, team=0)
    monkeypatch.setattr(user_interface, "perturb_throw", lambda _: released_throw)
    ui_state = user_interface.UIState()

    states = user_interface._apply_throw(ui_state, empty_board(1), throw)

    assert states.num_stones(0)[0] == 1
    assert ui_state.last_perturbation_offset_deg == 0.2
    assert ui_state.last_perturbation_probability == 0.25


def test_apply_throw_should_retain_no_perturbation_chance(monkeypatch):
    throw = Throw(angle_deg=0.0, speed=2.0, turn=0, y_val=2.5, team=0)
    monkeypatch.setattr(user_interface, "perturb_throw", lambda _: throw)
    ui_state = user_interface.UIState()

    user_interface._apply_throw(ui_state, empty_board(1), throw)

    assert ui_state.last_perturbation_offset_deg == 0.0
    assert ui_state.last_perturbation_probability == 0.5
