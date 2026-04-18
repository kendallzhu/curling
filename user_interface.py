import math
import numpy as np
import pygame
from dataclasses import dataclass

from constants import (
    starting_release_point,
    center_of_target_house,
    SHEET_W_M,
    SHEET_H_M,
    STONE_RADIUS_M,
    ROTATION_RATE,
)
from state import Throw

PANEL_H = 80  # pixels of control panel below the sheet
RED_TEAM_COLOR = (200, 50, 50)
YELLOW_TEAM_COLOR = (200, 200, 20)

TURN_LABELS = {1: "Clockwise", -1: "Counter", 0: "No Spin"}
TURN_COLOURS = {1: (160, 80, 80), -1: (80, 80, 160), 0: (80, 80, 80)}

def normalize(value, min, max):
    return (value - min) / (max - min)

def denormalize(normalized, min, max):
    return min + normalized * (max - min)

min_release_angle = -4
max_release_angle = 4

min_release_speed = 1.8
max_release_speed = 4.0

min_release_y = 2.25
max_release_y = 2.75

@dataclass
class UIState:
    angle_val: float = 0.0
    speed_val: float = 2.13
    y_val: float = 2.5
    turn_val: int = 0
    dragging_angle: bool = False
    dragging_speed: bool = False
    dragging_y: bool = False

    def to_next_throw(self, team: int) -> Throw:
        return Throw(
            angle_deg=self.angle_val,
            speed=self.speed_val,
            turn=self.turn_val,
            y_val=self.y_val,
            team=team,
        )


def render_sheet(surface: pygame.Surface, state) -> None:
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

def render_add_stone_preview(surface, throw: Throw):
    color = RED_TEAM_COLOR if throw.team == 0 else YELLOW_TEAM_COLOR
    sw, sh = surface.get_size()
    half_h = sh // 2
    scale = min(sw / (SHEET_W_M / 2), half_h / SHEET_H_M)
    ox = (sw - int((SHEET_W_M / 2) * scale)) // 2
    oy = 0

    def to_px(x_m, y_m):
        return int(x_m * scale) + ox, int(y_m * scale) + oy

    x, y = to_px(starting_release_point, throw.y_val)
    angle_rad = math.radians(throw.angle_deg)
    length = throw.speed * 1  # reduced scale
    dx = length * math.cos(angle_rad)
    dy = length * math.sin(angle_rad)
    end_x = x + int(dx * scale)
    end_y = y + int(dy * scale)

    # Draw arrow shaft
    pygame.draw.line(surface, color, (x, y), (end_x, end_y), 3)

    # Draw arrow head
    head_length = 0.15
    head_width = 0.1
    back_x = end_x - int(head_length * math.cos(angle_rad) * scale)
    back_y = end_y - int(head_length * math.sin(angle_rad) * scale)
    perp_x = int(head_width * (-math.sin(angle_rad)) * scale)
    perp_y = int(head_width * math.cos(angle_rad) * scale)
    left_x = back_x + perp_x
    left_y = back_y + perp_y
    right_x = back_x - perp_x
    right_y = back_y - perp_y
    pygame.draw.line(surface, color, (end_x, end_y), (left_x, left_y), 3)
    pygame.draw.line(surface, color, (end_x, end_y), (right_x, right_y), 3)

    # Curl indicator
    if throw.turn != 0:
        curl_angle = angle_rad + (math.pi / 2 if throw.turn == 1 else -math.pi / 2)
        curl_length = 0.3
        curl_dx = curl_length * math.cos(curl_angle)
        curl_dy = curl_length * math.sin(curl_angle)
        curl_end_x = end_x + int(curl_dx * scale)
        curl_end_y = end_y + int(curl_dy * scale)
        pygame.draw.line(surface, color, (end_x, end_y), (curl_end_x, curl_end_y), 2)
        # Small head for curl arrow
        curl_head_length = 0.1
        curl_back_x = curl_end_x - int(curl_head_length * math.cos(curl_angle) * scale)
        curl_back_y = curl_end_y - int(curl_head_length * math.sin(curl_angle) * scale)
        curl_perp_x = int(0.05 * (-math.sin(curl_angle)) * scale)
        curl_perp_y = int(0.05 * math.cos(curl_angle) * scale)
        curl_left_x = curl_back_x + curl_perp_x
        curl_left_y = curl_back_y + curl_perp_y
        curl_right_x = curl_back_x - curl_perp_x
        curl_right_y = curl_back_y - curl_perp_y
        pygame.draw.line(surface, color, (curl_end_x, curl_end_y), (curl_left_x, curl_left_y), 2)
        pygame.draw.line(surface, color, (curl_end_x, curl_end_y), (curl_right_x, curl_right_y), 2)

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
    speed_t = normalize(speed, min_release_speed, max_release_speed)
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
    y_t = normalize(y_val, min_release_y, max_release_y)
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


def render_ui(surface, ui_state: UIState, score, next_team: int):
    throw = ui_state.to_next_throw(next_team)
    render_add_stone_preview(surface, throw)
    return draw_panel(
        surface,
        ui_state.angle_val,
        ui_state.speed_val,
        ui_state.y_val,
        ui_state.turn_val,
        score,
    )


def add_stone(state, throw: Throw):
    angle_rad = math.radians(throw.angle_deg) + np.random.normal(0, 0.001)
    state.team = np.append(state.team, [[throw.team]], axis=1)
    state.x = np.append(state.x, [[starting_release_point]], axis=1)
    state.y = np.append(state.y, [[throw.y_val]], axis=1)
    speed = throw.speed + np.random.normal(0, 0.005)
    state.velocities.v = np.append(state.velocities.v, [[speed]], axis=1)
    state.velocities.theta = np.append(state.velocities.theta, [[angle_rad]], axis=1)
    state.rotation_directions = np.append(state.rotation_directions, [[throw.turn]], axis=1)

def handle_mouse_input(event, screen, ui_state, score, current_sheet_states):
    if event.type == pygame.MOUSEBUTTONDOWN:
        mx, my = event.pos
        btn_rect, (ax, ay, aw), (sx, sy, sw_), (yx, yy, yw), turn_rect = draw_panel(screen, ui_state.angle_val, ui_state.speed_val, ui_state.y_val, ui_state.turn_val, score)

        angle_knob_x = int(ax + normalize(ui_state.angle_val, min_release_angle, max_release_angle) * aw)
        speed_knob_x = int(sx + normalize(ui_state.speed_val, min_release_speed, max_release_speed) * sw_)
        y_knob_x = int(yx + normalize(ui_state.y_val, min_release_y, max_release_y) * yw)

        next_team_to_play = current_sheet_states.team_with_fewer_stones()

        if btn_rect.collidepoint(mx, my):
            throw = Throw(
                angle_deg=ui_state.angle_val,
                speed=ui_state.speed_val,
                turn=ui_state.turn_val,
                y_val=ui_state.y_val,
                team=next_team_to_play,
            )
            add_stone(current_sheet_states, throw)
        elif abs(mx - angle_knob_x) < 12 and abs(my - ay) < 12:
            ui_state.dragging_angle = True
        elif abs(mx - speed_knob_x) < 12 and abs(my - sy) < 12:
            ui_state.dragging_speed = True
        elif abs(mx - y_knob_x) < 12 and abs(my - yy) < 12:
            ui_state.dragging_y = True
        elif turn_rect.collidepoint(mx, my):
            ui_state.turn_val = {1: -1, -1: 0, 0: 1}[ui_state.turn_val]

    elif event.type == pygame.MOUSEBUTTONUP:
        ui_state.dragging_angle = False
        ui_state.dragging_speed = False
        ui_state.dragging_y = False

    elif event.type == pygame.MOUSEMOTION and (ui_state.dragging_angle or ui_state.dragging_speed or ui_state.dragging_y):
        mx, my = event.pos
        _, (ax, ay, aw), (sx, sy, sw_), (yx, yy, yw), _ = draw_panel(screen, ui_state.angle_val, ui_state.speed_val, ui_state.y_val, ui_state.turn_val, score)
        if ui_state.dragging_angle:
            t = max(0.0, min(1.0, (mx - ax) / aw))
            ui_state.angle_val = denormalize(t, min_release_angle, max_release_angle)
        if ui_state.dragging_speed:
            t = max(0.0, min(1.0, (mx - sx) / sw_))
            ui_state.speed_val = denormalize(t, min_release_speed, max_release_speed)
        if ui_state.dragging_y:
            t = max(0.0, min(1.0, (mx - yx) / yw))
            ui_state.y_val = denormalize(t, min_release_y, max_release_y)

    return ui_state