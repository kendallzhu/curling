import math
import os

starting_release_point = 11.2776
center_of_target_house = 39.624
house_outer_circle_radius = 1.8288

g = 9.8
mu = 0.0082
frac_pivot_time = 3.7e-4
frame_time = 0.1

SHEET_W_M = 45.0
SHEET_H_M = 5.0
center_line_y = SHEET_H_M / 2.0
STONE_RADIUS_M = 0.145
STONE_INNER_RING_RADIUS_M = 0.0625
ROTATION_RATE = (4 * 2 * math.pi) / 25.0  # 4 full rotations over 25m

# Sentinel x (and y) value marking a stone column as not currently in play
# (not yet thrown, or knocked out). Real thrown stones never reach x=0, which
# is behind starting_release_point.
NOT_IN_PLAY_X = 0.0

ui_sim_index = 0

# Throw parameter bounds (shared by UI sliders and bot search)
min_release_angle = -4.0
max_release_angle = 4.0
min_release_speed = 2.0
max_release_speed = 2.5
min_release_y = 2.25
max_release_y = 2.75
turn_options = (-1, 0, 1)

_repo_root = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.join(_repo_root, "weights")
datasets_dir = os.path.join(_repo_root, "datasets")

# Checked into git; overwrite after training via curling_nn.write_q_weights
q_network_weights_path = os.path.join(weights_dir, "q_network_weights.npz")
value_network_weights_path = os.path.join(weights_dir, "value_network_weights.npz")
q_network_datasets_dir = os.path.join(datasets_dir, "q_network")
value_network_datasets_dir = os.path.join(datasets_dir, "value_network")
