"""Experimental throw searchers used by scratch experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import bot
from constants import (
    max_release_angle,
    max_release_speed,
    max_release_y,
    min_release_angle,
    min_release_speed,
    min_release_y,
    turn_options,
)
from state import SheetStates, Throws, tile_sheet_states


class HierarchicalRandomThrows(bot.ThrowSearcher):
    """Search randomly, then refine the maximum-scoring candidates."""

    def __init__(
        self,
        rng: np.random.Generator,
        n_initial_throws: int,
        refinement_levels: Sequence[tuple[int, float]],
    ):
        if n_initial_throws < 1:
            raise ValueError("n_initial_throws must be positive")
        if not refinement_levels:
            raise ValueError("refinement_levels must not be empty")
        for num_throws, noise_scale in refinement_levels:
            if num_throws < 1:
                raise ValueError("Each refinement throw count must be positive")
            if noise_scale < 0:
                raise ValueError("Each noise scale must be non-negative")
        self.rng = rng
        self.n_initial_throws = n_initial_throws
        self.refinement_levels = tuple(refinement_levels)

    def _random_throws(self, team: int, count: int) -> Throws:
        return Throws(
            angle_deg=self.rng.uniform(min_release_angle, max_release_angle, size=count),
            speed=self.rng.uniform(min_release_speed, max_release_speed, size=count),
            turn=self.rng.choice(turn_options, size=count, replace=True),
            y_val=self.rng.uniform(min_release_y, max_release_y, size=count),
            team=np.full(count, team, dtype=int),
        )

    def get_throws(self, team: int) -> Throws:
        return self._random_throws(team, self.n_initial_throws)

    def get_throws_for_num_sims(
        self, *, team: int, sheet_states: SheetStates
    ) -> tuple[Throws, SheetStates]:
        num_sims = sheet_states.x.shape[0]
        current = self._random_throws(team, self.n_initial_throws * num_sims)
        angle_width = max_release_angle - min_release_angle
        speed_width = max_release_speed - min_release_speed
        y_width = max_release_y - min_release_y
        for num_refined_throws, noise_scale in self.refinement_levels:
            num_current_throws = current.angle_deg.size // num_sims
            current_states = tile_sheet_states(sheet_states, num_current_throws)
            current_scores = bot.score_throws_by_net_score(current_states, current).reshape(
                num_current_throws, num_sims
            )
            max_scores = current_scores.max(axis=0)
            refined = []
            for sim_idx in range(num_sims):
                best_indices = np.flatnonzero(
                    current_scores[:, sim_idx] == max_scores[sim_idx]
                )
                selected_indices = self.rng.choice(
                    best_indices, size=num_refined_throws, replace=True
                )
                current_indices = selected_indices * num_sims + sim_idx
                refined.append(
                    Throws(
                        angle_deg=np.clip(
                            current.angle_deg[current_indices]
                            + self.rng.normal(0, noise_scale * angle_width, num_refined_throws),
                            min_release_angle, max_release_angle,
                        ),
                        speed=np.clip(
                            current.speed[current_indices]
                            + self.rng.normal(0, noise_scale * speed_width, num_refined_throws),
                            min_release_speed, max_release_speed,
                        ),
                        turn=current.turn[current_indices],
                        y_val=np.clip(
                            current.y_val[current_indices]
                            + self.rng.normal(0, noise_scale * y_width, num_refined_throws),
                            min_release_y, max_release_y,
                        ),
                        team=np.full(num_refined_throws, team, dtype=int),
                    )
                )
            current = Throws(
                angle_deg=np.concatenate([t.angle_deg for t in refined]).reshape(num_sims, -1).T.ravel(),
                speed=np.concatenate([t.speed for t in refined]).reshape(num_sims, -1).T.ravel(),
                turn=np.concatenate([t.turn for t in refined]).reshape(num_sims, -1).T.ravel(),
                y_val=np.concatenate([t.y_val for t in refined]).reshape(num_sims, -1).T.ravel(),
                team=np.full(num_sims * num_refined_throws, team, dtype=int),
            )

        return current, tile_sheet_states(sheet_states, current.angle_deg.size // num_sims)


class RepulsiveHierarchicalRandomThrows(bot.ThrowSearcher):
    """Hierarchical random search with score-aware diversity selection."""

    def __init__(
        self,
        rng: np.random.Generator,
        n_initial_throws: int,
        level_configs: Sequence[tuple[int, int]],
        c1: float,
        c2: float,
    ):
        if n_initial_throws < 1:
            raise ValueError("n_initial_throws must be positive")
        if not level_configs:
            raise ValueError("level_configs must not be empty")
        if c1 < 0 or c2 < 0:
            raise ValueError("c1 and c2 must be non-negative")
        if any(generated < selected or selected < 1 for generated, selected in level_configs):
            raise ValueError("Each level must generate at least one selected throw")
        self.rng = rng
        self.n_initial_throws = n_initial_throws
        self.level_configs = tuple(level_configs)
        self.c1 = c1
        self.c2 = c2

    def _random_throws(self, team: int, count: int) -> Throws:
        return Throws(
            angle_deg=self.rng.uniform(min_release_angle, max_release_angle, count),
            speed=self.rng.uniform(min_release_speed, max_release_speed, count),
            turn=self.rng.choice(turn_options, count, replace=True),
            y_val=self.rng.uniform(min_release_y, max_release_y, count),
            team=np.full(count, team, dtype=int),
        )

    def get_throws(self, team: int) -> Throws:
        return self._random_throws(team, self.n_initial_throws)

    def get_throws_for_num_sims(
        self, *, team: int, sheet_states: SheetStates
    ) -> tuple[Throws, SheetStates]:
        num_sims = sheet_states.x.shape[0]
        existing = self._random_throws(team, self.n_initial_throws * num_sims)
        existing_states = tile_sheet_states(sheet_states, self.n_initial_throws)
        existing_scores = bot.score_throws_by_net_score(existing_states, existing).reshape(
            self.n_initial_throws, num_sims
        )
        angle_width = max_release_angle - min_release_angle
        speed_width = max_release_speed - min_release_speed
        y_width = max_release_y - min_release_y

        for num_generated, num_selected in self.level_configs:
            pool = self._random_throws(team, num_generated * num_sims)
            existing_count = existing.angle_deg.size // num_sims
            selected_by_state = []
            pool_cells = self._cell_ids(pool, num_generated, num_sims)
            existing_cells = self._cell_ids(existing, existing_count, num_sims)
            for sim_idx in range(num_sims):
                pool_indices = np.arange(num_generated) * num_sims + sim_idx
                existing_indices = np.arange(existing_count) * num_sims + sim_idx
                log_penalty = np.zeros(num_generated)
                for cell in range(64):
                    pool_local = np.flatnonzero(pool_cells[:, sim_idx] == cell)
                    existing_local = np.flatnonzero(existing_cells[:, sim_idx] == cell)
                    if pool_local.size == 0 or existing_local.size == 0:
                        continue
                    pool_cell_indices = pool_indices[pool_local]
                    existing_cell_indices = existing_indices[existing_local]
                    distance = np.sqrt(
                        ((pool.angle_deg[pool_cell_indices, None] - existing.angle_deg[existing_cell_indices]) / angle_width) ** 2
                        + ((pool.speed[pool_cell_indices, None] - existing.speed[existing_cell_indices]) / speed_width) ** 2
                        + ((pool.y_val[pool_cell_indices, None] - existing.y_val[existing_cell_indices]) / y_width) ** 2
                        + (pool.turn[pool_cell_indices, None] != existing.turn[existing_cell_indices])
                    )
                    badness = existing_scores[:, sim_idx].max() - existing_scores[existing_local, sim_idx]
                    log_penalty[pool_local] = np.mean(
                        -self.c1 * distance - self.c2 * badness[None, :], axis=1
                    )
                selected = np.argpartition(log_penalty, -num_selected)[-num_selected:]
                selected_by_state.append(pool_indices[selected])

            selected_indices = np.concatenate(selected_by_state).reshape(num_sims, num_selected).T.ravel()
            existing = Throws(
                angle_deg=pool.angle_deg[selected_indices],
                speed=pool.speed[selected_indices],
                turn=pool.turn[selected_indices],
                y_val=pool.y_val[selected_indices],
                team=np.full(num_sims * num_selected, team, dtype=int),
            )
            selected_states = tile_sheet_states(sheet_states, num_selected)
            existing_scores = bot.score_throws_by_net_score(selected_states, existing).reshape(
                num_selected, num_sims
            )

        return existing, tile_sheet_states(sheet_states, existing.angle_deg.size // num_sims)

    @staticmethod
    def _cell_ids(throws: Throws, num_throws: int, num_sims: int) -> np.ndarray:
        """Return 4x4x4 continuous-parameter cell IDs."""
        angle = np.clip(((throws.angle_deg - min_release_angle) / (max_release_angle - min_release_angle) * 4).astype(int), 0, 3).reshape(num_throws, num_sims)
        speed = np.clip(((throws.speed - min_release_speed) / (max_release_speed - min_release_speed) * 4).astype(int), 0, 3).reshape(num_throws, num_sims)
        y_val = np.clip(((throws.y_val - min_release_y) / (max_release_y - min_release_y) * 4).astype(int), 0, 3).reshape(num_throws, num_sims)
        return angle * 16 + speed * 4 + y_val
