import numpy as np
import pygame

from physics import run_sim, run_to_next_collision_or_stop
from scoring import get_score
from state import SheetStates, Velocities, empty_board, Throw
from user_interface import (
    render_sheet,
    render_ui,
    handle_mouse_input,
    PANEL_H,
    UIState,
)


if __name__ == "__main__":
    demo_collisions_sheet_states = SheetStates(
        team=np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]),
        x=np.array(
            [
                [
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    15.0,
                    15.5,
                    15.0,
                    15.5,
                    15.0,
                    15.5,
                    15.0,
                    15.5,
                ]
            ]
        ),
        y=np.array(
            [[2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.8, 1.5, 1, 2.4, 3.8, 4, 2.3, 3.2]]
        ),
        velocities=Velocities(
            v=np.array(
                [[2.5, 2.3, 2.8, 2.5, 2.2, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            theta=np.array(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
        ),
        rotation_directions=np.array([[1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]]),
    )

    guard_sheet_states = SheetStates(
        team=np.array([[0, 1]]),
        x=np.array([[36.6, 39.6]]),
        y=np.array([[2.5, 2.5]]),
        velocities=Velocities(
            v=np.array([[0.0, 0.0]]),
            theta=np.array([[0.0, 0.0]]),
        ),
        rotation_directions=np.array([[0, 0]]),
    )

    pygame.init()
    screen = pygame.display.set_mode((1800, 900 + PANEL_H), pygame.RESIZABLE)
    current_sheet_states = guard_sheet_states  # empty_board(1)
    timestep = 0.1

    # UI state
    ui_state = UIState()

    while True:
        next_team_to_play = current_sheet_states.team_with_fewer_stones()
        score = get_score(current_sheet_states)[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            ui_state, current_sheet_states = handle_mouse_input(
                event, screen, ui_state, score, current_sheet_states
            )

        render_sheet(screen, current_sheet_states.get_sheet(0))
        render_ui(screen, ui_state, score, next_team_to_play)
        pygame.display.flip()
        print(current_sheet_states)
        actual_timesteps, current_sheet_states = run_to_next_collision_or_stop(
            sheet_states=current_sheet_states, max_frame_time=np.inf
        )
        print(actual_timesteps)
        actual_timesteps = np.where(actual_timesteps == np.inf, 0.1, actual_timesteps)
        speedup = 5
        pygame.time.wait(int(actual_timesteps[0][0] * 1000) // speedup)

    pygame.quit()
