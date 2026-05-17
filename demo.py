import pygame
import time
import numpy as np
import constants
import bot
import copy

from physics import run_sim, run_to_next_collision_or_stop
from scoring import get_score
from presets import demo_collisions_sheet_states, guard_sheet_states, random_sheet_states
from state import empty_board
from user_interface import (
    render_sheet,
    render_ui,
    handle_mouse_input,
    PANEL_H,
    UIState,
)


class LagTracker:
    def __init__(self):
        self.last_print_time = time.time()
        self.total_lag = 0.0
        self.frame_count = 0
        self.total_intended_frame_time = 0.0
        self.print_interval_seconds = 5.0

    def add_lag(self, *, lag_ms: float, intended_frame_time_ms: int) -> None:
        self.total_lag += lag_ms
        self.frame_count += 1
        self.total_intended_frame_time += intended_frame_time_ms

    def maybe_print(self) -> None:
        current_time = time.time()
        if current_time - self.last_print_time < self.print_interval_seconds:
            return
        if self.frame_count > 0:
            avg_lag = self.total_lag / self.frame_count
            avg_intended_frame_time = self.total_intended_frame_time / self.frame_count
            print(f"Average lag: {avg_lag:.2f} ms over {self.frame_count} frames (average intended frame time: {avg_intended_frame_time:.2f} ms)")
        self.last_print_time = current_time
        self.total_lag = 0.0
        self.frame_count = 0
        self.total_intended_frame_time = 0.0


if __name__ == "__main__":
    pygame.init()
    monitor_size_multiplier = 1.8
    window_width = 1800 * monitor_size_multiplier
    window_height = window_width / 2 + PANEL_H
    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    previous_sheet_states = random_sheet_states(team1=4, team2=3) # guard_sheet_states()  # empty_board(1)
    timestep = 0.1

    # UI state
    ui_state = UIState()

    lag_tracker = LagTracker()
    has_state_changed = True

    while True:
        start_time = time.time()
        next_team_to_play = previous_sheet_states.team_with_fewer_stones()
        score = get_score(previous_sheet_states)[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            ui_state, next_sheet_states = handle_mouse_input(
                event,
                screen,
                ui_state,
                score,
                previous_sheet_states,
                preset_states=(
                    demo_collisions_sheet_states,
                    lambda: random_sheet_states(team1=4, team2=3),
                ),
            )

        render_sheet(screen, next_sheet_states.get_sheet(constants.ui_sim_index))
        render_ui(screen, ui_state, score, next_team_to_play)
        if has_state_changed and not(next_sheet_states.is_any_stone_moving()):
            bot_throw, bot_target_score, bot_robust_score = bot.get_throw_grid_search(next_sheet_states, next_team_to_play)
            print("Bot chosen throw:",bot_throw)
            print(f"Bot target score: {bot_target_score}, robust score: {bot_robust_score}")

        pygame.display.flip()

        actual_timesteps, next_sheet_states = run_to_next_collision_or_stop(
            sheet_states=copy.deepcopy(next_sheet_states), max_frame_time=0.03
        )
        has_state_changed = not(previous_sheet_states == next_sheet_states)
        previous_sheet_states = next_sheet_states

        # Waiting code below
        actual_timesteps = np.where(actual_timesteps == np.inf, 0.1, actual_timesteps)
        end_time = time.time()
        actual_time_ms = (end_time - start_time) * 1000
        speedup = 10
        intended_frame_time = int(actual_timesteps[constants.ui_sim_index].item() * 1000) // speedup
        if actual_time_ms > intended_frame_time:
            lag_tracker.add_lag(lag_ms=actual_time_ms - intended_frame_time, intended_frame_time_ms=intended_frame_time)
        #lag_tracker.maybe_print()
        wait_time = max(0, intended_frame_time - actual_time_ms)
        pygame.time.wait(int(wait_time))

    pygame.quit()
