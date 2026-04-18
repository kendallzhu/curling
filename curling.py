import numpy as np
import pygame

from physics import run_sim
from scoring import get_score
from state import SheetStates, Velocities, empty_board
from user_interface import (
    render_sheet,
    render_add_stone_preview,
    draw_panel,
    handle_mouse_input,
    PANEL_H,
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
    y_val = 2.5
    angle_val = 0.0  # degrees
    speed_val = 2.13  # m/s, 1.8 to 4.0
    turn_val = 0  # 1, -1, or 0
    dragging_angle = False
    dragging_speed = False
    dragging_y = False

    while True:
        next_team_to_play = current_sheet_states.team_with_fewer_stones()
        score = get_score(current_sheet_states)[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            angle_val, speed_val, y_val, turn_val, dragging_angle, dragging_speed, dragging_y = handle_mouse_input(event, screen, angle_val, speed_val, y_val, turn_val, score, current_sheet_states, dragging_angle, dragging_speed, dragging_y)

        render_sheet(screen, current_sheet_states)
        render_add_stone_preview(screen, angle_val, speed_val, turn_val, y_val, next_team_to_play)
        draw_panel(screen, angle_val, speed_val, y_val, turn_val, score)
        pygame.display.flip()
        actual_timesteps, current_sheet_states = run_sim(
            sheet_states=current_sheet_states, max_frame_time=timestep
        )
        speedup = 10
        pygame.time.wait(int(actual_timesteps[0][0] * 1000) // speedup)

    pygame.quit()
