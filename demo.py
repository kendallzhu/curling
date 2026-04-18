import pygame
import time
import numpy as np

from physics import run_sim
from scoring import get_score
from presets import demo_collisions_sheet_states, guard_sheet_states
from state import empty_board
from user_interface import (
    render_sheet,
    render_ui,
    handle_mouse_input,
    PANEL_H,
    UIState,
)


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1800, 900 + PANEL_H), pygame.RESIZABLE)
    current_sheet_states = guard_sheet_states()  # empty_board(1)
    timestep = 0.1

    # UI state
    ui_state = UIState()
    sim_index = 0

    # Lag tracking
    last_print_time = time.time()
    total_lag = 0.0
    frame_count = 0

    while True:
        start_time = time.time()
        next_team_to_play = current_sheet_states.team_with_fewer_stones()
        score = get_score(current_sheet_states)[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            ui_state, current_sheet_states = handle_mouse_input(
                event,
                screen,
                ui_state,
                score,
                current_sheet_states,
                preset_states=(demo_collisions_sheet_states, guard_sheet_states),
            )

        render_sheet(screen, current_sheet_states.get_sheet(sim_index))
        render_ui(screen, ui_state, score, next_team_to_play)
        pygame.display.flip()
        actual_timesteps, current_sheet_states = run_sim(
            sheet_states=current_sheet_states, max_frame_time=timestep
        )
        end_time = time.time()
        actual_time_ms = (end_time - start_time) * 1000
        speedup = 10
        intended_frame_time = int(actual_timesteps[sim_index].item() * 1000) // speedup
        if actual_time_ms > intended_frame_time:
            lag = actual_time_ms - intended_frame_time
            total_lag += lag
            frame_count += 1
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            if frame_count > 0:
                avg_lag = total_lag / frame_count
                print(f"Average lag: {avg_lag:.2f} ms over {frame_count} frames")
            last_print_time = current_time
            total_lag = 0.0
            frame_count = 0
        wait_time = max(0, intended_frame_time - actual_time_ms)
        pygame.time.wait(int(wait_time))

    pygame.quit()
