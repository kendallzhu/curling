import pygame
import time
import sys
import numpy as np
import constants
import bot
import copy
import curling_nn
import threading

from physics import run_to_next_collision_or_stop
from scoring import get_score
from presets import (
    demo_collisions_sheet_states,
    guard_sheet_states,
    random_sheet_states,
)
from state import empty_board
from user_interface import (
    render_sheet,
    render_ui,
    handle_mouse_input,
    PANEL_H,
    UIState,
)


def _should_quit(event) -> bool:
    return event.type == pygame.QUIT or (
        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
    )


def _quit_demo() -> None:
    pygame.quit()
    sys.exit(0)


def _run_bot_suggestions(sheet_states, team, value_network, value_normalizer):
    bot_throw, bot_target_score, bot_robust_score = bot.get_throw_grid_search(
        sheet_states, team
    )
    bot_throw_nn, bot_nn_expected_score = bot.get_throw_nn_argmax(
        sheet_states,
        team,
        neural_network=value_network,
        normalizer=value_normalizer,
    )
    return (
        bot_throw,
        bot_target_score,
        bot_robust_score,
        bot_throw_nn,
        bot_nn_expected_score,
    )


def _compute_bot_suggestions_interruptibly(
    sheet_states, team, value_network, value_normalizer
):
    """Run bot search off the UI thread so quit events are still handled."""
    result = {}
    error = {}

    def worker():
        try:
            result["value"] = _run_bot_suggestions(
                sheet_states, team, value_network, value_normalizer
            )
        except Exception as exc:  # noqa: BLE001 - surface to main thread
            error["exc"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        for event in pygame.event.get():
            if _should_quit(event):
                _quit_demo()
        pygame.event.pump()
        pygame.time.wait(50)
    if error:
        raise error["exc"]
    return result["value"]



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
            print(
                f"Average lag: {avg_lag:.2f} ms over {self.frame_count} frames (average intended frame time: {avg_intended_frame_time:.2f} ms)"
            )
        self.last_print_time = current_time
        self.total_lag = 0.0
        self.frame_count = 0
        self.total_intended_frame_time = 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Curling simulator demo")
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use a smaller window that fits a standard laptop display",
    )
    args = parser.parse_args()

    pygame.init()
    monitor_size_multiplier = 0.7 if args.small else 1.8
    window_width = 1800 * monitor_size_multiplier
    window_height = window_width / 2 + PANEL_H
    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    previous_sheet_states = random_sheet_states(
        team1=5, team2=4
    )  # guard_sheet_states()  # empty_board(1)
    timestep = 0.1

    ui_state = UIState()
    value_network, value_normalizer = curling_nn.load_weights(
        constants.value_network_weights_path
    )

    next_team_to_play = 1

    ui_state.bot_throw = None
    ui_state.bot_throw_nn = None

    lag_tracker = LagTracker()
    has_state_changed = True

    while True:
        start_time = time.time()
        score = get_score(previous_sheet_states)[0]
        next_sheet_states = previous_sheet_states

        for event in pygame.event.get():
            if _should_quit(event):
                _quit_demo()
            ui_state, next_sheet_states = handle_mouse_input(
                event,
                screen,
                ui_state,
                score,
                previous_sheet_states,
                preset_states=(
                    demo_collisions_sheet_states,
                    lambda: random_sheet_states(team1=5, team2=4),
                ),
            )
            if (
                next_sheet_states.is_any_stone_moving()
                and not previous_sheet_states.is_any_stone_moving()
            ):
                break

        render_sheet(screen, next_sheet_states.get_sheet(constants.ui_sim_index))
        render_ui(screen, ui_state, score, next_team_to_play)
        if has_state_changed and not (next_sheet_states.is_any_stone_moving()):
            next_team_to_play = (
                next_team_to_play
                if previous_sheet_states.num_stones(next_team_to_play)
                < previous_sheet_states.num_stones(1 - next_team_to_play)
                else 1 - next_team_to_play
            )
            pygame.display.flip()
            (
                bot_throw,
                bot_target_score,
                bot_robust_score,
                bot_throw_nn,
                bot_nn_expected_score,
            ) = _compute_bot_suggestions_interruptibly(
                next_sheet_states,
                next_team_to_play,
                value_network,
                value_normalizer,
            )
            print("Bot chosen throw:", bot_throw)
            print(
                f"Bot target score: {bot_target_score}, robust score: {bot_robust_score}"
            )
            ui_state.bot_throw = bot_throw

            print("Bot NN chosen throw:", bot_throw_nn)
            if bot_nn_expected_score is not None:
                print(f"Bot NN expected score: {bot_nn_expected_score}")
            ui_state.bot_throw_nn = bot_throw_nn

        pygame.display.flip()

        max_frame_time = 0.15
        actual_timesteps, next_sheet_states = run_to_next_collision_or_stop(
            sheet_states=copy.deepcopy(next_sheet_states), max_frame_time=max_frame_time
        )
        has_state_changed = not (previous_sheet_states == next_sheet_states)
        previous_sheet_states = next_sheet_states

        # Waiting code below
        actual_timesteps = np.where(
            actual_timesteps == np.inf, max_frame_time, actual_timesteps
        )
        end_time = time.time()
        actual_time_ms = (end_time - start_time) * 1000
        speedup = 5
        intended_frame_time = (
            int(actual_timesteps[constants.ui_sim_index].item() * 1000) // speedup
        )
        if actual_time_ms > intended_frame_time:
            lag_tracker.add_lag(
                lag_ms=actual_time_ms - intended_frame_time,
                intended_frame_time_ms=intended_frame_time,
            )
        # lag_tracker.maybe_print()
        wait_time = max(0, intended_frame_time - actual_time_ms)
        pygame.time.wait(int(wait_time))

    pygame.quit()
