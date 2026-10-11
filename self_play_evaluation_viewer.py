"""Replay paired self-play evaluation games in the demo's Pygame renderer."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pygame
import numpy as np

import self_play_training
from physics import run_to_next_collision_or_stop
from state import add_new_stone
from user_interface import render_sheet


LARGE_CHECKPOINT = Path("weights/self_play/round9_20k/checkpoints/m_30.npz")
DEFAULT_MODEL_VERSION = "round9_2k_v2"
GROUPS = (
    ("model_turn_7", "Model played turn 7", (7,)),
    ("model_turn_9", "Model played turn 9", (9,)),
    ("model_turn_8", "Model played turn 8", (8,)),
)
SIDEBAR_W = 430
ROW_H = 24
LIST_TOP = 400
BG = (31, 35, 43)
PANEL = (43, 49, 59)
TEXT = (235, 238, 242)
MUTED = (165, 174, 185)
ACCENT = (89, 137, 196)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay games from the paired self-play evaluation."
    )
    parser.add_argument("--num-states", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grid-size", type=int, nargs=3, default=(10, 10, 4))
    parser.add_argument("--max-stones", type=int, default=10)
    parser.add_argument("--large-checkpoint", type=Path, default=LARGE_CHECKPOINT)
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help="Self-play run name; by default the viewer uses its latest checkpoint.",
    )
    parser.add_argument(
        "--small-checkpoint",
        type=Path,
        help="Optional checkpoint path override for --model-version.",
    )
    parser.add_argument(
        "--small-window", action="store_true", help="Use a smaller starting window."
    )
    return parser.parse_args()


def _latest_checkpoint(model_version: str) -> Path:
    checkpoint_dir = Path("weights/self_play") / model_version / "checkpoints"
    checkpoints = []
    for path in checkpoint_dir.glob("m_*.npz"):
        try:
            checkpoint_index = int(path.stem.removeprefix("m_"))
        except ValueError:
            continue
        checkpoints.append((checkpoint_index, path))
    if not checkpoints:
        raise FileNotFoundError(
            f"No model checkpoints found for {model_version}: {checkpoint_dir}"
        )
    return max(checkpoints)[1]


def _button(surface, rect, label, font, *, active=False):
    pygame.draw.rect(
        surface, ACCENT if active else (65, 73, 86), rect, border_radius=5
    )
    surface.blit(font.render(label, True, TEXT), (rect.x + 9, rect.y + 7))


def _selected_record(records, run_name, group_index):
    group_name = GROUPS[group_index][0]
    return next(
        record
        for record in records
        if record["run"] == run_name and record["group"] == group_name
    )


def _summary_stats(record):
    games = record["games"]
    model_scores = np.asarray([game["model_score"] for game in games], dtype=float)
    greedy_scores = np.asarray([game["greedy_score"] for game in games], dtype=float)
    differences = model_scores - greedy_scores
    average_advantage = float(np.mean(differences))
    standard_error = (
        float(np.std(differences, ddof=1) / np.sqrt(differences.size))
        if differences.size > 1
        else float("nan")
    )
    return {
        "num_games": int(differences.size),
        "average_model_score": float(np.mean(model_scores)),
        "average_greedy_score": float(np.mean(greedy_scores)),
        "average_advantage": average_advantage,
        "advantage_ci95_low": average_advantage - 1.96 * standard_error,
        "advantage_ci95_high": average_advantage + 1.96 * standard_error,
        "win_rate": float(np.mean(differences > 0)),
        "tie_rate": float(np.mean(differences == 0)),
        "loss_rate": float(np.mean(differences < 0)),
    }


def _run_viewer(records, *, run_names: tuple[str, str], small_window: bool) -> None:
    pygame.init()
    window_w, window_h = (1400, 760) if small_window else (1880, 1000)
    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    pygame.display.set_caption("Self-play evaluation games")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    small_font = pygame.font.SysFont(None, 19)
    title_font = pygame.font.SysFont(None, 27, bold=True)

    run_index = 0
    group_index = 0
    game_index = 0
    scroll_offset = 0
    greedy_vs_greedy = False
    playing = False
    current = None
    throw_index = 0

    def reset_game():
        nonlocal current, throw_index, playing
        game = _selected_record(records, run_names[run_index], group_index)["games"][game_index]
        current = copy.deepcopy(game["initial_state"])
        throw_index = 0
        playing = False

    def keep_game_visible():
        nonlocal scroll_offset
        visible_rows = max(1, (screen.get_height() - LIST_TOP - 10) // ROW_H)
        if game_index < scroll_offset:
            scroll_offset = game_index
        elif game_index >= scroll_offset + visible_rows:
            scroll_offset = game_index - visible_rows + 1

    reset_game()

    running = True
    while running:
        screen_w, screen_h = screen.get_size()
        sidebar_x = max(0, screen_w - SIDEBAR_W)
        sheet_surface = screen.subsurface(pygame.Rect(0, 0, sidebar_x, screen_h))
        record = _selected_record(records, run_names[run_index], group_index)
        game = record["games"][game_index]
        replay_throws = game["greedy_throws"] if greedy_vs_greedy else game["model_throws"]
        model_score = game["model_score"]
        greedy_score = game["greedy_score"]
        model_turn_text = ", ".join(str(turn) for turn in record["model_turns"])
        summary = _summary_stats(record)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if throw_index >= len(replay_throws) and not current.is_any_stone_moving():
                        reset_game()
                    playing = not playing
                elif event.key in (pygame.K_RIGHT, pygame.K_DOWN):
                    game_index = min(game_index + 1, len(record["games"]) - 1)
                    keep_game_visible()
                    reset_game()
                elif event.key in (pygame.K_LEFT, pygame.K_UP):
                    game_index = max(0, game_index - 1)
                    keep_game_visible()
                    reset_game()
                elif event.key == pygame.K_g:
                    greedy_vs_greedy = not greedy_vs_greedy
                    reset_game()
                elif event.key == pygame.K_r:
                    reset_game()
                elif event.key == pygame.K_PAGEUP:
                    scroll_offset = max(0, scroll_offset - 10)
                elif event.key == pygame.K_PAGEDOWN:
                    scroll_offset = min(max(0, len(record["games"]) - 1), scroll_offset + 10)
            elif event.type == pygame.MOUSEWHEEL:
                scroll_offset = max(
                    0,
                    min(len(record["games"]) - 1, scroll_offset - event.y * 3),
                )
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button in (4, 5) and mx >= sidebar_x:
                    scroll_offset = max(
                        0,
                        min(len(record["games"]) - 1, scroll_offset + (1 if event.button == 5 else -1) * 3),
                    )
                elif mx >= sidebar_x:
                    local_x = mx - sidebar_x
                    if 42 <= my < 72:
                        if local_x < 202:
                            run_index = (run_index - 1) % len(run_names)
                        else:
                            run_index = (run_index + 1) % len(run_names)
                        game_index = min(game_index, len(_selected_record(records, run_names[run_index], group_index)["games"]) - 1)
                        reset_game()
                    elif 78 <= my < 108:
                        if local_x < 202:
                            group_index = (group_index - 1) % len(GROUPS)
                        else:
                            group_index = (group_index + 1) % len(GROUPS)
                        game_index = 0
                        scroll_offset = 0
                        reset_game()
                    elif 218 <= my < 250:
                        if local_x < 166:
                            greedy_vs_greedy = not greedy_vs_greedy
                            reset_game()
                        elif local_x < 274:
                            if throw_index >= len(replay_throws) and not current.is_any_stone_moving():
                                reset_game()
                            playing = not playing
                        else:
                            reset_game()
                    elif my >= LIST_TOP:
                        clicked_index = scroll_offset + (my - LIST_TOP) // ROW_H
                        if clicked_index < len(record["games"]):
                            game_index = clicked_index
                            reset_game()

        record = _selected_record(records, run_names[run_index], group_index)
        game = record["games"][game_index]
        replay_throws = game["greedy_throws"] if greedy_vs_greedy else game["model_throws"]
        model_score = game["model_score"]
        greedy_score = game["greedy_score"]
        model_turn_text = ", ".join(str(turn) for turn in record["model_turns"])
        summary = _summary_stats(record)

        # Advance one collision/stop interval per frame, as in demo.py.
        if playing:
            if not current.is_any_stone_moving():
                if throw_index == len(replay_throws):
                    playing = False
                else:
                    current = add_new_stone(current, replay_throws[throw_index])
                    throw_index += 1
            if current.is_any_stone_moving():
                _, current = run_to_next_collision_or_stop(
                    sheet_states=copy.deepcopy(current), max_frame_time=0.1
                )

        render_sheet(sheet_surface, current.get_sheet(0))
        screen.fill(BG, pygame.Rect(sidebar_x, 0, screen_w - sidebar_x, screen_h))
        pygame.draw.line(screen, (77, 87, 100), (sidebar_x, 0), (sidebar_x, screen_h), 2)
        panel_x = sidebar_x + 12
        panel_w = screen_w - sidebar_x - 24

        screen.blit(title_font.render("Evaluation games", True, TEXT), (panel_x, 12))
        _button(screen, pygame.Rect(panel_x, 42, 190, 30), "‹ Run", font)
        _button(screen, pygame.Rect(panel_x + 198, 42, panel_w - 198, 30), "Run ›", font)
        screen.blit(small_font.render(record["run"], True, TEXT), (panel_x + 72, 49))
        _button(screen, pygame.Rect(panel_x, 78, 190, 30), "‹ Turn group", font)
        _button(screen, pygame.Rect(panel_x + 198, 78, panel_w - 198, 30), "Turn group ›", font)
        screen.blit(small_font.render("Turns " + model_turn_text, True, TEXT), (panel_x + 70, 85))

        _button(
            screen,
            pygame.Rect(panel_x, 218, 154, 32),
            "Greedy vs Greedy" if greedy_vs_greedy else "Model vs Greedy",
            small_font,
            active=True,
        )
        _button(
            screen,
            pygame.Rect(panel_x + 162, 218, 92, 32),
            "Pause" if playing else "Play",
            font,
        )
        _button(screen, pygame.Rect(panel_x + 266, 218, panel_w - 266, 32), "Reset", font)

        pygame.draw.rect(screen, PANEL, pygame.Rect(panel_x, 116, panel_w, 94), border_radius=5)
        screen.blit(
            font.render(f"Paired summary · {summary['num_games']} games", True, TEXT),
            (panel_x + 10, 121),
        )
        screen.blit(
            small_font.render(
                f"Model avg {summary['average_model_score']:+.2f}  |  Greedy avg {summary['average_greedy_score']:+.2f}",
                True,
                TEXT,
            ),
            (panel_x + 10, 145),
        )
        screen.blit(
            small_font.render(
                f"Advantage {summary['average_advantage']:+.2f}  |  95% CI [{summary['advantage_ci95_low']:+.2f}, {summary['advantage_ci95_high']:+.2f}]",
                True,
                TEXT,
            ),
            (panel_x + 10, 165),
        )
        screen.blit(
            small_font.render(
                f"Win / tie / loss: {summary['win_rate']:.0%} / {summary['tie_rate']:.0%} / {summary['loss_rate']:.0%}",
                True,
                TEXT,
            ),
            (panel_x + 10, 185),
        )

        advantage = model_score - greedy_score
        result = "W" if advantage > 0 else "L" if advantage < 0 else "T"
        pygame.draw.rect(screen, PANEL, pygame.Rect(panel_x, 258, panel_w, 112), border_radius=5)
        screen.blit(
            font.render(f"Game {game_index + 1} of {len(record['games'])}  |  model turns {model_turn_text}", True, TEXT),
            (panel_x + 10, 266),
        )
        screen.blit(
            small_font.render(
                f"Model {model_score:+.0f}   Greedy {greedy_score:+.0f}   Paired result: {result} ({advantage:+.0f})",
                True,
                TEXT,
            ),
            (panel_x + 10, 292),
        )
        stone_moving = current.is_any_stone_moving()
        if stone_moving:
            active_throw_index = throw_index - 1
            throw_state_label = "Throwing now"
        elif throw_index < len(replay_throws):
            active_throw_index = throw_index
            throw_state_label = "Next throw"
        else:
            active_throw_index = max(0, len(replay_throws) - 1)
            throw_state_label = "Game complete"
        active_turn = 7 + active_throw_index
        active_throw = replay_throws[active_throw_index]
        team_name = "Red" if active_throw.team == 0 else "Yellow"
        model_turn = active_turn in record["model_turns"]
        model_is_doing_throw = model_turn and not greedy_vs_greedy
        policy_name = "Model" if model_is_doing_throw else "Greedy"

        screen.blit(
            small_font.render(
                f"{throw_state_label}: turn {active_turn} · {team_name} team",
                True,
                MUTED,
            ),
            (panel_x + 10, 314),
        )
        screen.blit(
            small_font.render(
                f"Team has model turn: {'Yes' if model_turn else 'No'}  |  Current policy: {policy_name}",
                True,
                MUTED,
            ),
            (panel_x + 10, 336),
        )

        screen.blit(
            small_font.render("#  TURN  MODEL  GREEDY  RESULT  ·  scroll/click", True, MUTED),
            (panel_x + 4, 376),
        )
        list_bottom = screen_h - 10
        pygame.draw.rect(
            screen,
            (25, 29, 36),
            pygame.Rect(panel_x, LIST_TOP, panel_w, max(0, list_bottom - LIST_TOP)),
            border_radius=4,
        )
        previous_clip = screen.get_clip()
        screen.set_clip(pygame.Rect(panel_x, LIST_TOP, panel_w, max(0, list_bottom - LIST_TOP)))
        visible_rows = max(0, (list_bottom - LIST_TOP + ROW_H - 1) // ROW_H)
        for row_index in range(scroll_offset, min(len(record["games"]), scroll_offset + visible_rows)):
            row_y = LIST_TOP + (row_index - scroll_offset) * ROW_H
            row_game = record["games"][row_index]
            difference = row_game["model_score"] - row_game["greedy_score"]
            row_result = "W" if difference > 0 else "L" if difference < 0 else "T"
            if row_index == game_index:
                pygame.draw.rect(screen, (54, 83, 119), (panel_x + 1, row_y, panel_w - 2, ROW_H))
            elif row_index % 2 == 0:
                pygame.draw.rect(screen, (34, 39, 47), (panel_x + 1, row_y, panel_w - 2, ROW_H))
            turn_label = ",".join(str(turn) for turn in record["model_turns"])
            row_text = (
                f"{row_index + 1:03}       {turn_label:<5}       "
                f"{row_game['model_score']:+3.0f}       {row_game['greedy_score']:+3.0f}       {row_result}"
            )
            screen.blit(small_font.render(row_text, True, TEXT), (panel_x + 7, row_y + 3))
        screen.set_clip(previous_clip)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main() -> None:
    args = _parse_args()
    small_checkpoint = args.small_checkpoint or _latest_checkpoint(args.model_version)
    run_names = ("large_m30", args.model_version)
    checkpoint_paths = {
        run_names[0]: args.large_checkpoint,
        run_names[1]: small_checkpoint,
    }
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    print("Recomputing the seeded paired evaluation and retaining game throws...")
    records = self_play_training.evaluate_checkpoints_against_greedy_with_games(
        checkpoint_paths=checkpoint_paths,
        num_states=args.num_states,
        seed=args.seed,
        grid_size=tuple(args.grid_size),
        max_stones=args.max_stones,
    )
    print(f"Ready to replay {sum(len(record['games']) for record in records)} games.")
    _run_viewer(records, run_names=run_names, small_window=args.small_window)


if __name__ == "__main__":
    main()
