import math
import numpy as np
import pygame
from dataclasses import dataclass


@dataclass
class Velocities:
 v: np.array # (num_sims, num_stones)
 theta: np.array # (num_sims, num_stones)


@dataclass
class SheetStates:
   team: np.array # (num_sims, num_stones) 0/1
   x: np.array # (num_sims, num_stones)
   y: np.array # (num_sims, num_stones)
   velocities: Velocities
   rotation_directions: np.array # (num_sims, num_stones) 0/-1/1

   def num_stones(self, of_team):
       return np.sum(self.team[0] == of_team)
   
   def team_with_fewer_stones(self):
       return 0 if self.num_stones(0) < self.num_stones(1) else 1


@dataclass
class VelocityHistories:
 v: np.array # (num_sims, num_stones, num_timesteps)
 theta: np.array # (num_sims, num_stones, num_timesteps)


@dataclass
class SheetHistories:
   t: np.array # (num_sims, num_timesteps)
   x: np.array # (num_sims, num_stones, num_timesteps)
   y: np.array # (num_sims, num_stones, num_timesteps)
   velocities: VelocityHistories
   rotation_directions: np.array # (num_sims, num_stones) 0/-1/1

def empty_board(num_sims: int) -> SheetStates:
   return SheetStates(
       x=np.zeros((num_sims, 0)),
       y=np.zeros((num_sims, 0)),
       rotation_directions=np.zeros((num_sims, 0)),
       velocities=Velocities(v=np.zeros((num_sims, 0)), theta=np.zeros((num_sims, 0))),
       team=np.zeros((num_sims, 0))
   )


starting_release_point = 11.2776
center_of_target_house = 39.624
house_outer_circle_radius = 1.8288

def get_score(sheet_states: SheetStates) -> np.array:  # (num_sims, 2)
   num_sims = sheet_states.x.shape[0]
   distance_from_center = np.sqrt(
       (sheet_states.x - center_of_target_house) ** 2 + (sheet_states.y - 2.5)**2
   )
   team_scores = np.zeros((num_sims, 2), dtype=int)

   if distance_from_center.shape[1] == 0:
       return team_scores

   in_house = distance_from_center < house_outer_circle_radius + R
   team_closest_stone_in_house = np.ones((num_sims, 2), dtype=int) * np.inf

   for i in range(2):
       team_closest_stone_in_house[:, i] = np.min(
           np.where((sheet_states.team == i) & in_house, distance_from_center, np.inf),
           axis=1,
       )
   for i in range(2):
       team_scores[:, i] = (
           in_house & (distance_from_center < team_closest_stone_in_house[:, [1 - i]])
       ).sum(axis=1)
   return team_scores

R = 0.14
r = 0.0625
g = 9.8
mu = 0.0082
frac_pivot_time = 3.7e-4
frame_time = 0.1
r_p = (R * R / 2 + r * r) ** 0.5



def get_collision_times(*, x1, y1, v1, theta1, x2, y2, v2, theta2, R):
   xvel_diff = v1 * np.cos(theta1) - v2 * np.cos(theta2)
   yvel_diff = v1 * np.sin(theta1) - v2 * np.sin(theta2)
   xpos_diff = x1 - x2
   ypos_diff = y1 - y2
   a = xvel_diff**2 + yvel_diff**2
   b = 2 * (xpos_diff * xvel_diff + ypos_diff * yvel_diff)
   c = xpos_diff**2 + ypos_diff**2 - 4 * R**2


   d = b**2 - 4 * a * c
   sol_exists = d >= 0
   x = np.sqrt(np.fmax(d, 0))
   return np.where(
       (~sol_exists) | (a == 0),
       np.inf,
       np.where(
           -b - x >= 0,
           (-b - x) / (2 * a),
           np.where(-b + x >= 0, (-b + x) / (2 * a), np.inf),
       ),
   )

def apply_collision(*, x1, y1, v1, theta1, x2, y2, v2, theta2):
   phi = np.arctan2(y2 - y1, x2 - x1)
   new_xvel1 = v2 * np.cos(phi) * np.cos(theta2 - phi) - v1 * np.sin(phi) * np.sin(
       theta1 - phi
   )
   new_yvel1 = v2 * np.sin(phi) * np.cos(theta2 - phi) + v1 * np.cos(phi) * np.sin(
       theta1 - phi
   )
   new_xvel2 = v1 * np.cos(phi) * np.cos(theta1 - phi) - v2 * np.sin(phi) * np.sin(
       theta2 - phi
   )
   new_yvel2 = v1 * np.sin(phi) * np.cos(theta1 - phi) + v2 * np.cos(phi) * np.sin(
       theta2 - phi
   )
   return {
       "v1": np.sqrt(new_xvel1**2 + new_yvel1**2),
       "theta1": np.arctan2(new_yvel1, new_xvel1),
       "v2": np.sqrt(new_xvel2**2 + new_yvel2**2),
       "theta2": np.arctan2(new_yvel2, new_xvel2),
   }

def run_sim(
   *, sheet_states: SheetStates, max_frame_time: float
) -> tuple[np.array, SheetStates]:
   v = sheet_states.velocities.v
   theta = sheet_states.velocities.theta
   team=sheet_states.team
   x = sheet_states.x
   y = sheet_states.y
   num_sims, num_stones = v.shape

   if team.shape[1] == 0:
       return np.zeros((num_sims, 1)), sheet_states

   time_to_stop = v / (mu * g)
   next_stop_time = np.min(np.where(v > 0, time_to_stop, np.inf), axis=1)
   next_collision_time = np.ones(num_sims) * np.inf
   first_collision_index = np.zeros(num_sims, dtype=int)
   second_collision_index = np.zeros(num_sims, dtype=int)
   for i in range(num_stones):
       for j in range(i):
           # this .99 stops it from getting stuck
           time_to_collision = (
               get_collision_times(
                   x1=x[:, j],
                   y1=y[:, j],
                   v1=v[:, j],
                   theta1=theta[:, j],
                   x2=x[:, i],
                   y2=y[:, i],
                   v2=v[:, i],
                   theta2=theta[:, i],
                   R=R,
               )
               * 0.99
           )
           first_collision_index = np.where(
               time_to_collision < next_collision_time, j, first_collision_index
           )
           second_collision_index = np.where(
               time_to_collision < next_collision_time, i, second_collision_index
           )
           next_collision_time = np.fmin(next_collision_time, time_to_collision)
   next_event_is_collision = next_collision_time <= np.fmin(
       next_stop_time, max_frame_time
   )
   time_to_next_event = np.fmin(
       next_collision_time, np.fmin(next_stop_time, max_frame_time)
   ).reshape((num_sims, 1))
   x += np.cos(theta) * v * time_to_next_event
   y += np.sin(theta) * v * time_to_next_event
   theta += (
       sheet_states.rotation_directions
       / r_p
       * v
       * frac_pivot_time
       * time_to_next_event
   )
   v = np.fmax(v - mu * g * time_to_next_event, 0)
   sims_with_collisions = np.arange(num_sims, dtype=int)[next_event_is_collision]
   idx1 = first_collision_index[sims_with_collisions]
   idx2 = second_collision_index[sims_with_collisions]
   collision_results = apply_collision(
       x1=x[sims_with_collisions, idx1],
       y1=y[sims_with_collisions, idx1],
       v1=v[sims_with_collisions, idx1],
       theta1=theta[sims_with_collisions, idx1],
       x2=x[sims_with_collisions, idx2],
       y2=y[sims_with_collisions, idx2],
       v2=v[sims_with_collisions, idx2],
       theta2=theta[sims_with_collisions, idx2],
   )
   v[sims_with_collisions, idx1] = collision_results["v1"]
   theta[sims_with_collisions, idx1] = collision_results["theta1"]
   v[sims_with_collisions, idx2] = collision_results["v2"]
   theta[sims_with_collisions, idx2] = collision_results["theta2"]


   return time_to_next_event, SheetStates(
       team=team,
       x=x,
       y=y,
       velocities=Velocities(
           v=v,
           theta=theta,
       ),
       rotation_directions=sheet_states.rotation_directions,
   )

PANEL_H = 80  # pixels of control panel below the sheet
SHEET_W_M = 45.0
SHEET_H_M = 5.0
STONE_RADIUS_M = 0.145
ROTATION_RATE = (4 * 2 * math.pi) / 25.0  # 4 full rotations over 25m
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
        pygame.draw.rect(surface, (200, 220, 255),
                         (ox, oy, int(half_w * scale), int(SHEET_H_M * scale)))

        # Hog lines
        for x_m in (10.0, 35.0):
            if x_offset_m <= x_m <= x_offset_m + half_w:
                x_px = ox + int((x_m - x_offset_m) * scale)
                pygame.draw.line(surface, (220, 60, 60),
                                 (x_px, oy), (x_px, oy + int(SHEET_H_M * scale)), 2)

        # Houses
        for tee_x in (5.0, center_of_target_house):
            if x_offset_m <= tee_x <= x_offset_m + half_w:
                tee_px = ox + int((tee_x - x_offset_m) * scale)
                for radius_m, colour in [
                    (1.83, (60,  100, 220)),
                    (1.22, (255, 255, 255)),
                    (0.61, (220, 80,  80)),
                    (0.15, (255, 255, 255)),
                ]:
                    pygame.draw.circle(surface, colour, (tee_px, mid_y),
                                       int(radius_m * scale))
                pygame.draw.line(surface, (60, 60, 220),
                                 (tee_px, oy), (tee_px, oy + int(SHEET_H_M * scale)), 1)

        # Stones
        r = max(2, int(STONE_RADIUS_M * scale))
        xs, ys, dirs, teams = state.x[0], state.y[0], state.rotation_directions[0], state.team[0]

        for x, y, d, team in zip(xs, ys, dirs, teams):
            assert(team == 0 or team == 1)
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
                pygame.draw.line(surface, (60, 60, 60), (hx1, hy1), (hx2, hy2), max(2, r // 3))

    surface.fill((160, 180, 210))
    draw_half(0.0, 0)
    draw_half(SHEET_W_M / 2, half_h)

TURN_LABELS = {1: "Clockwise", -1: "Counter", 0: "No Spin"}
TURN_COLOURS = {1: (160, 80, 80), -1: (80, 80, 160), 0: (80, 80, 80)}

def normalize(value, min, max):
    return (value - min) / (max - min)

def denormalize(normalized, min, max):
    return min + normalized * (max - min)

min_release_angle = -3
max_release_angle = 3
def draw_panel(surface, angle, speed, y_val, turn_val, score):
    sw, sh = surface.get_size()
    panel_y = sh - PANEL_H
    pygame.draw.rect(surface, (40, 40, 40), (0, panel_y, sw, PANEL_H))

    font = pygame.font.SysFont(None, 24)

    # Button
    btn_rect = pygame.Rect(20, panel_y + 20, 120, 40)
    pygame.draw.rect(surface, (80, 160, 80), btn_rect, border_radius=6)
    surface.blit(font.render("Add Stone", True, (255, 255, 255)),
                 (btn_rect.x + 10, btn_rect.y + 10))

    # Angle slider
    angle_x = 200
    slider_y = panel_y + 40
    slider_w = 200
    pygame.draw.rect(surface, (100, 100, 100), (angle_x, slider_y - 4, slider_w, 8), border_radius=4)
    angle_t = normalize(angle, min_release_angle, max_release_angle)
    pygame.draw.circle(surface, (200, 200, 200),
                       (int(angle_x + angle_t * slider_w), slider_y), 10)
    surface.blit(font.render(f"Angle: {angle:.2f}°", True, (200, 200, 200)),
                 (angle_x, panel_y + 12))

    # Speed slider
    speed_x = 480
    pygame.draw.rect(surface, (100, 100, 100), (speed_x, slider_y - 4, slider_w, 8), border_radius=4)
    speed_t = (speed - 1.8) / (4.0 - 1.8)  # 0..1
    pygame.draw.circle(surface, (200, 200, 200),
                       (int(speed_x + speed_t * slider_w), slider_y), 10)
    surface.blit(font.render(f"Speed: {speed:.2f} m/s", True, (200, 200, 200)),
                 (speed_x, panel_y + 12))
    
    # Y offset slider
    y_x = 760
    pygame.draw.rect(surface, (100, 100, 100), (y_x, slider_y - 4, slider_w, 8), border_radius=4)
    y_t = (y_val - 2.5 + 0.25) / 0.5  # 0..1
    pygame.draw.circle(surface, (200, 200, 200),
                       (int(y_x + y_t * slider_w), slider_y), 10)
    surface.blit(font.render(f"Y: {y_val:.2f} m", True, (200, 200, 200)),
                 (y_x, panel_y + 12))
    
    # Turn toggle
    turn_x = 990
    turn_rect = pygame.Rect(turn_x, panel_y + 20, 120, 40)
    pygame.draw.rect(surface, TURN_COLOURS[turn_val], turn_rect, border_radius=6)
    surface.blit(font.render(TURN_LABELS[turn_val], True, (255, 255, 255)),
                 (turn_rect.x + 8, turn_rect.y + 10))
    
    # Score display
    s0, s1 = int(score[0]), int(score[1])
    score_text = font.render(f"Red  {s0}  —  {s1}  Yellow", True, (220, 220, 220))
    surface.blit(score_text, (sw - 260, panel_y + 28))

    return btn_rect, (angle_x, slider_y, slider_w), (speed_x, slider_y, slider_w), (y_x, slider_y, slider_w), turn_rect

def add_stone(state, angle_deg, speed, turn, y_val, team):
    angle_rad = math.radians(angle_deg) + np.random.normal(0, 0.001)
    state.team = np.append(state.team, [[team]], axis=1)
    state.x = np.append(state.x, [[starting_release_point]], axis=1)
    state.y = np.append(state.y, [[y_val]],  axis=1)
    speed = speed + np.random.normal(0, .005)
    state.velocities.v     = np.append(state.velocities.v,     [[speed]], axis=1)
    state.velocities.theta = np.append(state.velocities.theta, [[angle_rad]], axis=1)
    state.rotation_directions = np.append(state.rotation_directions, [[turn]], axis=1)

if __name__ == "__main__":
    demo_collisions_sheet_states = SheetStates(
    team=np.array([[0, 1, 0, 1, 0, 1,  0, 1, 0, 1, 0, 1, 0, 1]]),
    x=np.array([[0.5, 1.0, 1.5, 2.0, 2.5, 3.0,  15.0, 15.5, 15.0, 15.5, 15.0, 15.5, 15.0, 15.5]]),
    y=np.array([[2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.8,  1.5,  1,  2.4,  3.8,  4,  2.3,  3.2]]),
    velocities=Velocities(
            v=    np.array([[2.5, 2.3, 2.8, 2.5, 2.2, 3.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]]),
            theta=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]]),
        ),
        rotation_directions=np.array([[1, -1, 1, -1, 1, -1,   0,    0,    0,    0,    0,    0,    0,    0]]),
    )

    guard_sheet_states = SheetStates(
        team=np.array([[0, 1]]),
        x=np.array([[36.6, 39.6]]),
        y=np.array([[2.5, 2.5]]),
        velocities=Velocities(
                v=    np.array([[0.0, 0.0]]),
                theta=np.array([[0.0, 0.0]]),
            ),
            rotation_directions=np.array([[0, 0]]),
    )

    pygame.init()
    screen = pygame.display.set_mode((1800, 900 + PANEL_H), pygame.RESIZABLE)
    current_sheet_states = empty_board(1)
    timestep = 0.1

    # UI state
    y_val = 2.5
    angle_val = 0.0    # degrees
    speed_val = 2.13    # m/s, 1.8 to 4.0
    turn_val = 0  # 1, -1, or 0
    dragging_angle = False
    dragging_speed = False
    dragging_y=False
    
    while True:
        next_team_to_play = current_sheet_states.team_with_fewer_stones()
        score = get_score(current_sheet_states)[0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                btn_rect, (ax, ay, aw), (sx, sy, sw_), (yx, yy, yw), turn_rect = draw_panel(screen, angle_val, speed_val, y_val, turn_val, score)

                if btn_rect.collidepoint(mx, my):
                    add_stone(current_sheet_states, angle_val, speed_val, turn_val, y_val, team=next_team_to_play)
                elif abs(mx - int(ax + ((angle_val - min_release_angle) / (max_release_angle - min_release_angle)) * aw)) < 12 and abs(my - ay) < 12:
                    dragging_angle = True
                elif abs(mx - int(sx + ((speed_val - 1.8) / 2.2) * sw_)) < 12 and abs(my - sy) < 12:
                    dragging_speed = True
                elif abs(mx - int(yx + ((y_val - 2.25) / 0.5) * yw)) < 12 and abs(my - yy) < 12:
                    dragging_y = True
                elif turn_rect.collidepoint(mx, my):
                    turn_val = {1: -1, -1: 0, 0: 1}[turn_val]  # cycle through


            if event.type == pygame.MOUSEBUTTONUP:
                dragging_angle = False
                dragging_speed = False
                dragging_y = False

            if event.type == pygame.MOUSEMOTION and (dragging_angle or dragging_speed or dragging_y):
                mx, my = event.pos
                _, (ax, ay, aw), (sx, sy, sw_), (yx, yy, yw), _ = draw_panel(screen, angle_val, speed_val, y_val, turn_val, score)
                if dragging_angle:
                    t = max(0.0, min(1.0, (mx - ax) / aw))
                    angle_val = denormalize(t, min_release_angle, max_release_angle)
                if dragging_speed:
                    t = max(0.0, min(1.0, (mx - sx) / sw_))
                    speed_val = 1.8 + t * 2.2
                if dragging_y:
                    t = max(0.0, min(1.0, (mx - yx) / yw))
                    y_val = 2.25 + t * .5

        render_sheet(screen, current_sheet_states)
        draw_panel(screen, angle_val, speed_val, y_val, turn_val, score)
        pygame.display.flip()
        actual_timesteps, current_sheet_states = run_sim(sheet_states=current_sheet_states, max_frame_time=timestep)
        speedup = 10
        pygame.time.wait(int(actual_timesteps[0][0] * 1000) // speedup)

    pygame.quit()