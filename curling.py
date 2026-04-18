import math
import numpy as np
import pygame

from constants import (
    starting_release_point,
    center_of_target_house,
    house_outer_circle_radius,
    SHEET_W_M,
    SHEET_H_M,
    STONE_RADIUS_M,
    ROTATION_RATE,
)

from physics import run_sim
from scoring import get_score
from state import SheetStates, Velocities

PANEL_H = 80  # pixels of control panel below the sheet
RED_TEAM_COLOR = (200, 50, 50)
YELLOW_TEAM_COLOR = (200, 200, 20)


def render_sheet(surface: pygame.Surface, state: SheetStates) -> None:
    sw, sh = surface.get_size()
    half_h = sh // 2
    scale = min(sw / (SHEET_W_M / 2), half_h / SHEET_H_M)

    def draw_half(x_offset_m: float, screen_oy: int) -> None:
        ox = (sw - int((SHEET_W_M / 2) * scale)) // 2
        oy = screen_oy
        mid_y = oy + int(SHEET_H_M * scale) // 2
        half_w = SHEET_W_M / 2

        def to_px(x_m, y_m):
            return int((x_m - x_offset_m) * scale) + ox, int(y_m * scale) + oy

        # Sheet
        pygame.draw.rect(
            surface,
            (200, 220, 255),
            (ox, oy, int(half_w * scale), int(SHEET_H_M * scale)),
        )

        # Hog lines
        for x_m in (10.0, 35.0):
            if x_offset_m <= x_m <= x_offset_m + half_w:
                x_px = ox + int((x_m - x_offset_m) * scale)
                pygame.draw.line(
                    surface,
                    (220, 60, 60),
                    (x_px, oy),
                    (x_px, oy + int(SHEET_H_M * scale)),
                    2,
                )

        # Houses
        for tee_x in (5.0, center_of_target_house):
            if x_offset_m <= tee_x <= x_offset_m + half_w:
                tee_px = ox + int((tee_x - x_offset_m) * scale)
                for radius_m, colour in [
                    (1.83, (60, 100, 220)),
                    (1.22, (255, 255, 255)),
                    (0.61, (220, 80, 80)),
                    (0.15, (255, 255, 255)),
                ]:
                    pygame.draw.circle(
                        surface, colour, (tee_px, mid_y), int(radius_m * scale)
                    )
                pygame.draw.line(
                    surface,
                    (60, 60, 220),
                    (tee_px, oy),
                    (tee_px, oy + int(SHEET_H_M * scale)),
                    1,
                )

        # Stones
        r = max(2, int(STONE_RADIUS_M * scale))
        xs, ys, dirs, teams = (
            state.x[0],
            state.y[0],
            state.rotation_directions[0],
            state.team[0],
        )

        for x, y, d, team in zip(xs, ys, dirs, teams):
            assert team == 0 or team == 1
            color = RED_TEAM_COLOR if team == 0 else YELLOW_TEAM_COLOR
            if x_offset_m <= x <= x_offset_m + half_w:
                cx, cy = to_px(float(x), float(y))
                pygame.draw.circle(surface, color, (cx, cy), r)
                pygame.draw.circle(surface, (80, 80, 80), (cx, cy), r, max(1, r // 3))

                angle = d * ROTATION_RATE * x
                hx2 = cx + int(math.cos(angle) * r * 1)
                hy2 = cy + int(math.sin(angle) * r * 1)
                hx1 = cx - int(math.cos(angle) * r * 0.3)
                hy1 = cy - int(math.sin(angle) * r * 0.3)
                pygame.draw.line(
                    surface, (60, 60, 60), (hx1, hy1), (hx2, hy2), max(2, r // 3)
                )

    surface.fill((160, 180, 210))
    draw_half(0.0, 0)
    draw_half(SHEET_W_M / 2, half_h)


TURN_LABELS = {1: "Clockwise", -1: "Counter", 0: "No Spin"}
TURN_COLOURS = {1: (160, 80, 80), -1: (80, 80, 160), 0: (80, 80, 80)}


def normalize(value, min, max):
    return (value - min) / (max - min)


def denormalize(normalized, min, max):
    return min + normalized * (max - min)


min_release_angle = -4
max_release_angle = 4


def draw_panel(surface, angle, speed, y_val, turn_val, score):
    sw, sh = surface.get_size()
    panel_y = sh - PANEL_H
    pygame.draw.rect(surface, (40, 40, 40), (0, panel_y, sw, PANEL_H))

    font = pygame.font.SysFont(None, 24)

    # Button
    btn_rect = pygame.Rect(20, panel_y + 20, 120, 40)
    pygame.draw.rect(surface, (80, 160, 80), btn_rect, border_radius=6)
    surface.blit(
        font.render("Add Stone", True, (255, 255, 255)),
        (btn_rect.x + 10, btn_rect.y + 10),
    )

    # Angle slider
    angle_x = 200
    slider_y = panel_y + 40
    slider_w = 200
    pygame.draw.rect(
        surface, (100, 100, 100), (angle_x, slider_y - 4, slider_w, 8), border_radius=4
    )
    angle_t = normalize(angle, min_release_angle, max_release_angle)
    pygame.draw.circle(
        surface, (200, 200, 200), (int(angle_x + angle_t * slider_w), slider_y), 10
    )
    surface.blit(
        font.render(f"Angle: {angle:.2f}°", True, (200, 200, 200)),
        (angle_x, panel_y + 12),
    )

    # Speed slider
    speed_x = 480
    pygame.draw.rect(
        surface, (100, 100, 100), (speed_x, slider_y - 4, slider_w, 8), border_radius=4
    )
    speed_t = (speed - 1.8) / (4.0 - 1.8)  # 0..1
    pygame.draw.circle(
        surface, (200, 200, 200), (int(speed_x + speed_t * slider_w), slider_y), 10
    )
    surface.blit(
        font.render(f"Speed: {speed:.2f} m/s", True, (200, 200, 200)),
        (speed_x, panel_y + 12),
    )

    # Y offset slider
    y_x = 760
    pygame.draw.rect(
        surface, (100, 100, 100), (y_x, slider_y - 4, slider_w, 8), border_radius=4
    )
    y_t = (y_val - 2.5 + 0.25) / 0.5  # 0..1
    pygame.draw.circle(
        surface, (200, 200, 200), (int(y_x + y_t * slider_w), slider_y), 10
    )
    surface.blit(
        font.render(f"Y: {y_val:.2f} m", True, (200, 200, 200)), (y_x, panel_y + 12)
    )

    # Turn toggle
    turn_x = 990
    turn_rect = pygame.Rect(turn_x, panel_y + 20, 120, 40)
    pygame.draw.rect(surface, TURN_COLOURS[turn_val], turn_rect, border_radius=6)
    surface.blit(
        font.render(TURN_LABELS[turn_val], True, (255, 255, 255)),
        (turn_rect.x + 8, turn_rect.y + 10),
    )

    # Score display
    s0, s1 = int(score[0]), int(score[1])
    score_text = font.render(f"Red  {s0}  —  {s1}  Yellow", True, (220, 220, 220))
    surface.blit(score_text, (sw - 260, panel_y + 28))

    return (
        btn_rect,
        (angle_x, slider_y, slider_w),
        (speed_x, slider_y, slider_w),
        (y_x, slider_y, slider_w),
        turn_rect,
    )


def add_stone(state, angle_deg, speed, turn, y_val, team):
    angle_rad = math.radians(angle_deg) + np.random.normal(0, 0.001)
    state.team = np.append(state.team, [[team]], axis=1)
    state.x = np.append(state.x, [[starting_release_point]], axis=1)
    state.y = np.append(state.y, [[y_val]], axis=1)
    speed = speed + np.random.normal(0, 0.005)
    state.velocities.v = np.append(state.velocities.v, [[speed]], axis=1)
    state.velocities.theta = np.append(state.velocities.theta, [[angle_rad]], axis=1)
    state.rotation_directions = np.append(state.rotation_directions, [[turn]], axis=1)


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

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                btn_rect, (ax, ay, aw), (sx, sy, sw_), (yx, yy, yw), turn_rect = (
                    draw_panel(screen, angle_val, speed_val, y_val, turn_val, score)
                )

                if btn_rect.collidepoint(mx, my):
                    add_stone(
                        current_sheet_states,
                        angle_val,
                        speed_val,
                        turn_val,
                        y_val,
                        team=next_team_to_play,
                    )
                elif (
                    abs(
                        mx
                        - int(
                            ax
                            + (
                                (angle_val - min_release_angle)
                                / (max_release_angle - min_release_angle)
                            )
                            * aw
                        )
                    )
                    < 12
                    and abs(my - ay) < 12
                ):
                    dragging_angle = True
                elif (
                    abs(mx - int(sx + ((speed_val - 1.8) / 2.2) * sw_)) < 12
                    and abs(my - sy) < 12
                ):
                    dragging_speed = True
                elif (
                    abs(mx - int(yx + ((y_val - 2.25) / 0.5) * yw)) < 12
                    and abs(my - yy) < 12
                ):
                    dragging_y = True
                elif turn_rect.collidepoint(mx, my):
                    turn_val = {1: -1, -1: 0, 0: 1}[turn_val]  # cycle through

            if event.type == pygame.MOUSEBUTTONUP:
                dragging_angle = False
                dragging_speed = False
                dragging_y = False

            if event.type == pygame.MOUSEMOTION and (
                dragging_angle or dragging_speed or dragging_y
            ):
                mx, my = event.pos
                _, (ax, ay, aw), (sx, sy, sw_), (yx, yy, yw), _ = draw_panel(
                    screen, angle_val, speed_val, y_val, turn_val, score
                )
                if dragging_angle:
                    t = max(0.0, min(1.0, (mx - ax) / aw))
                    angle_val = denormalize(t, min_release_angle, max_release_angle)
                if dragging_speed:
                    t = max(0.0, min(1.0, (mx - sx) / sw_))
                    speed_val = 1.8 + t * 2.2
                if dragging_y:
                    t = max(0.0, min(1.0, (mx - yx) / yw))
                    y_val = 2.25 + t * 0.5

        render_sheet(screen, current_sheet_states)
        draw_panel(screen, angle_val, speed_val, y_val, turn_val, score)
        pygame.display.flip()
        actual_timesteps, current_sheet_states = run_sim(
            sheet_states=current_sheet_states, max_frame_time=timestep
        )
        speedup = 10
        pygame.time.wait(int(actual_timesteps[0][0] * 1000) // speedup)

    pygame.quit()
