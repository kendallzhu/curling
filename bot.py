import numpy as np

from state import Throw, SheetStates, add_new_stone_from_throw
import scoring
import physics

def get_throw(state: SheetStates, team) -> Throw:
    return Throw(
        angle_deg=0,
        speed=2.13,
        turn=0,
        y_val=2.5,
        team=team,
    )

def simulate_score_after_throw(state: SheetStates, throw: Throw) -> np.ndarray: # (num_sims, 1)
    new_state = add_new_stone_from_throw(state, throw)
    final_state = physics.run_until_stopping_fast(sheet_states=new_state)
    score = scoring.get_net_score_for_team(final_state, throw.team)
    return score
