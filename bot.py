import numpy as np
import math
from collections.abc import Sequence
from typing import Callable, Protocol

from state import (
    Throw,
    Throws,
    SheetStates,
    Velocities,
    add_new_stone,
    add_new_stones,
    add_stones_from_throws,
    add_noise_to_throw,
    tile_sheet_states,
    empty_board,
)

from dataset import Normalizer

from constants import (
    min_release_angle,
    max_release_angle,
    min_release_speed,
    max_release_speed,
    min_release_y,
    max_release_y,
    turn_options,
    q_network_weights_path,
    value_network_weights_path,
)

import nn
import scoring
import physics
import curling_nn


def get_throw(state: SheetStates, team) -> Throw:
    return Throw(
        angle_deg=0,
        speed=2.13,
        turn=0,
        y_val=2.5,
        team=team,
    )


def simulate_score_after_throw(
    state: SheetStates, throw: Throw
) -> np.ndarray:  # (num_sims, 1)
    new_state = add_new_stone(state, throw)
    final_state = physics.run_until_stopping(sheet_states=new_state)
    score = scoring.get_net_score_for_team(final_state, throw.team)
    return score


def simulate_average_scores_with_noise(
    state: SheetStates, throws: list[Throw], num_samples: int = 20
) -> np.ndarray:
    # for each throw, generate num_samples noisy versions
    noisy_throws = [add_noise_to_throw(t) for t in throws for _ in range(num_samples)]
    noisy_state = tile_sheet_states(state, len(noisy_throws))
    noisy_state = add_new_stones(noisy_state, noisy_throws)
    final_state = physics.run_until_stopping(sheet_states=noisy_state)
    scores = scoring.get_net_score_for_team(
        final_state, throws[0].team
    )  # (len(throws) * num_samples,)
    # reshape and average over samples for each throw
    return scores.reshape(len(throws), num_samples).mean(axis=1)  # (len(throws),)


class CurlingPolicy(Protocol):
    def make_throws(
        self, sheet_states: SheetStates, team: int, rng: np.random.Generator
    ) -> Throws: ...


class ThrowSearcher(Protocol):
    def get_throws(self, team: int) -> Throws: ...

    def get_throws_for_num_sims(self, *, team: int, sheet_states: SheetStates) -> tuple[Throws, SheetStates]: ...

class ThrowsGridSearcher(ThrowSearcher):
    def __init__(self, num_angles: int, num_speeds: int, num_y_vals: int):
        self.num_angles = num_angles
        self.num_speeds = num_speeds
        self.num_y_vals = num_y_vals

    def get_throws(self, team: int):
        angle_options = np.linspace(min_release_angle, max_release_angle, self.num_angles)
        speed_options = np.linspace(min_release_speed, max_release_speed, self.num_speeds)
        y_options = np.linspace(min_release_y, max_release_y, self.num_y_vals)

        angles, speeds, turns, ys = np.meshgrid(
            angle_options, speed_options, turn_options, y_options, indexing='ij'
        )

        return Throws(
            angle_deg=angles.flatten(),
            speed=speeds.flatten(),
            turn=turns.flatten(),
            y_val=ys.flatten(),
            team=np.ones(angles.size, dtype=int) * team
        )

    def get_throws_for_num_sims(self, *, team: int, sheet_states: SheetStates) -> tuple[Throws, SheetStates]:
        num_sims = sheet_states.x.shape[0]
        angle_options = np.linspace(min_release_angle, max_release_angle, self.num_angles)
        speed_options = np.linspace(min_release_speed, max_release_speed, self.num_speeds)
        y_options = np.linspace(min_release_y, max_release_y, self.num_y_vals)

        angles, speeds, turns, ys = np.meshgrid(
            angle_options, speed_options, turn_options, y_options, indexing='ij'
        )

        angles_flat = angles.flatten()
        speeds_flat = speeds.flatten()
        turns_flat = turns.flatten()
        ys_flat = ys.flatten()
        n_throws = len(angles_flat)

        angle_deg = np.repeat(angles_flat, num_sims)
        speed = np.repeat(speeds_flat, num_sims)
        turn = np.repeat(turns_flat, num_sims)
        y_val = np.repeat(ys_flat, num_sims)

        throws = Throws(
            angle_deg=angle_deg,
            speed=speed,
            turn=turn,
            y_val=y_val,
            team=np.ones(angle_deg.size, dtype=int) * team
        )

        tiled_sheet_states = tile_sheet_states(sheet_states, n_throws)
        return throws, tiled_sheet_states


def get_most_robust_throw_with_score(
    *,
    state: SheetStates,
    throws: list[Throw],
    scores: np.ndarray,
    target_score: float,
    max_throws_to_evaluate: int,
) -> tuple[Throw, float]:
    best_throws = [
        throw for throw, score in zip(throws, scores) if score == target_score
    ]
    assert (
        len(best_throws) > 0
    )  # if we get to min score, we should find something robust for it
    print(
        f"Found {len(best_throws)} throws with max score {target_score}, evaluating robustness..."
    )
    if len(best_throws) > max_throws_to_evaluate:
        print(
            f"Evaluating robustness for {max_throws_to_evaluate} randomly selected throws out of {len(best_throws)}"
        )
        indices = np.random.choice(
            len(best_throws), size=max_throws_to_evaluate, replace=False
        )
        best_throws = [best_throws[i] for i in indices]

    robust_scores = simulate_average_scores_with_noise(state, best_throws)
    max_robust_score = np.max(robust_scores)
    best_idx = int(np.random.choice(np.where(robust_scores == max_robust_score)[0]))
    return best_throws[best_idx], max_robust_score


def get_throw_q_argmax(
    state: SheetStates,
    team: int,
    *,
    seed: int = 0,
    throw_searcher: ThrowSearcher | None = None,
    neural_network: curling_nn.QNetwork | None = None,
    normalizer: Normalizer | None = None,
) -> tuple[Throw | None, float | None]:
    """Suggest a throw via ArgmaxThrowPolicy.from_q_network.

    Returns (None, None) if the board stone count does not match the network.
    The float is the Q-network expected net score for the throwing team.
    """
    if throw_searcher is None:
        throw_searcher = ThrowsGridSearcher(num_angles=20, num_speeds=20, num_y_vals=6)
    if neural_network is None or normalizer is None:
        neural_network, normalizer = curling_nn.load_q_weights(
            q_network_weights_path
        )
    if state.x.shape[1] != neural_network.num_stones:
        return None, None
    policy = ArgmaxThrowPolicy.from_q_network(
        random_action_prob=0.0,
        neural_network=neural_network,
        normalizer=normalizer,
        throw_searcher=throw_searcher,
    )
    throws = policy.make_throws(state, team, np.random.default_rng(seed))
    throw = Throw(
        angle_deg=float(throws.angle_deg[0]),
        speed=float(throws.speed[0]),
        turn=int(throws.turn[0]),
        y_val=float(throws.y_val[0]),
        team=int(throws.team[0]),
    )
    expected_score = float(
        policy.scoring_function(
            state,
            Throws(
                angle_deg=throws.angle_deg[:1],
                speed=throws.speed[:1],
                turn=throws.turn[:1],
                y_val=throws.y_val[:1],
                team=throws.team[:1],
            ),
        )[0]
    )
    return throw, expected_score


def get_throw_v_argmax(
    state: SheetStates,
    team: int,
    *,
    seed: int = 0,
    throw_searcher: ThrowSearcher | None = None,
    neural_network: curling_nn.ValueNetwork | None = None,
    normalizer: Normalizer | None = None,
) -> tuple[Throw | None, float | None]:
    """Suggest a throw by simulating candidates and scoring the result with V.

    Enabled only when the board is one stone short of the value network's
    ``num_stones`` (second-to-last throw). Returns (None, None) otherwise.
    The float is V's expected net score for the throwing team after the throw.
    """
    if throw_searcher is None:
        throw_searcher = ThrowsGridSearcher(num_angles=20, num_speeds=20, num_y_vals=6)
    if neural_network is None or normalizer is None:
        neural_network, normalizer = curling_nn.load_v_weights(
            value_network_weights_path
        )
    if state.x.shape[1] + 1 != neural_network.num_stones:
        return None, None
    policy = ArgmaxThrowPolicy.from_value_network(
        random_action_prob=0.0,
        neural_network=neural_network,
        normalizer=normalizer,
        throw_searcher=throw_searcher,
    )
    throws = policy.make_throws(state, team, np.random.default_rng(seed))
    throw = Throw(
        angle_deg=float(throws.angle_deg[0]),
        speed=float(throws.speed[0]),
        turn=int(throws.turn[0]),
        y_val=float(throws.y_val[0]),
        team=int(throws.team[0]),
    )
    expected_score = float(
        policy.scoring_function(
            state,
            Throws(
                angle_deg=throws.angle_deg[:1],
                speed=throws.speed[:1],
                turn=throws.turn[:1],
                y_val=throws.y_val[:1],
                team=throws.team[:1],
            ),
        )[0]
    )
    return throw, expected_score


def get_throw_grid_search(state: SheetStates, team: int) -> tuple[Throw, float, float]:
    candidate_throws = ThrowsGridSearcher(
        num_angles=20, num_speeds=20, num_y_vals=6
    ).get_throws(team)
    num_combos = candidate_throws.angle_deg.shape[0]
    print(f"Grid search: evaluating {num_combos} throws")

    throws = [
        Throw(
            angle_deg=float(candidate_throws.angle_deg[i]),
            speed=float(candidate_throws.speed[i]),
            turn=int(candidate_throws.turn[i]),
            y_val=float(candidate_throws.y_val[i]),
            team=team,
        )
        for i in range(num_combos)
    ]
    tiled_state = tile_sheet_states(state, num_combos)
    tiled_state = add_new_stones(tiled_state, throws)

    final_state = physics.run_until_stopping(sheet_states=tiled_state)
    scores = scoring.get_net_score_for_team(final_state, team)  # (num_combos,)

    target_score = np.max(scores)
    max_throws_to_evaluate = num_combos // 20
    while True:
        best_throw, robust_score = get_most_robust_throw_with_score(
            state=state,
            throws=throws,
            scores=scores,
            target_score=target_score,
            max_throws_to_evaluate=max_throws_to_evaluate,
        )
        if robust_score >= target_score - 1:
            return best_throw, target_score, robust_score
        target_score -= 1


class RandomThrows(ThrowSearcher):
    def __init__(self, rng: np.random.Generator, n_throws_to_generate: int):
        self.rng = rng
        self.n_throws_to_generate = n_throws_to_generate

    def get_throws(self, team: int):
        return Throws(
            angle_deg=self.rng.uniform(
                min_release_angle, max_release_angle, size=self.n_throws_to_generate
            ),
            speed=self.rng.uniform(
                min_release_speed, max_release_speed, size=self.n_throws_to_generate
            ),
            turn=self.rng.choice(
                turn_options, size=self.n_throws_to_generate, replace=True
            ),
            y_val=self.rng.uniform(
                min_release_y, max_release_y, size=self.n_throws_to_generate
            ),
            team=np.ones(self.n_throws_to_generate, dtype=int) * team,
        )

    def get_throws_for_num_sims(self, *, team: int, sheet_states: SheetStates) -> tuple[Throws, SheetStates]:
        num_sims = sheet_states.x.shape[0]
        total_throws = num_sims * self.n_throws_to_generate
        throws = Throws(
            angle_deg=self.rng.uniform(
                min_release_angle, max_release_angle, size=total_throws
            ),
            speed=self.rng.uniform(
                min_release_speed, max_release_speed, size=total_throws
            ),
            turn=self.rng.choice(turn_options, size=total_throws, replace=True),
            y_val=self.rng.uniform(min_release_y, max_release_y, size=total_throws),
            team=np.ones(total_throws, dtype=int) * team,
        )
        tiled_sheet_states = tile_sheet_states(sheet_states, self.n_throws_to_generate)
        return throws, tiled_sheet_states


class HierarchicalRandomThrows(ThrowSearcher):
    """Search randomly, then refine the maximum-scoring candidates.

    The first ``n_initial_throws`` candidates are scored independently for each
    sheet.  At each refinement level, candidates sample with replacement from
    that sheet's maximum-score candidates and add clipped Gaussian noise to
    the continuous release parameters.  The generated candidates are scored
    before the next level, so each level narrows the search around the best
    candidates from the previous level. Turns remain discrete and are
    inherited from the selected candidate.

    ``noise_scale`` is expressed as a fraction of each parameter's configured
    search-range width, so the same value works across angle, speed, and
    release-y.
    """

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
        # This fallback is useful for callers that only need a candidate batch;
        # state-aware refinement happens in get_throws_for_num_sims.
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
            current_scores = score_throws_by_net_score(current_states, current).reshape(
                num_current_throws, num_sims
            )
            max_scores = current_scores.max(axis=0)
            refined = []
            for sim_idx in range(num_sims):
                # Keep all ties, rather than choosing one representative, so
                # the next level can explore every equally good basin.
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
                            + self.rng.normal(
                                0,
                                noise_scale * angle_width,
                                num_refined_throws,
                            ),
                            min_release_angle,
                            max_release_angle,
                        ),
                        speed=np.clip(
                            current.speed[current_indices]
                            + self.rng.normal(
                                0,
                                noise_scale * speed_width,
                                num_refined_throws,
                            ),
                            min_release_speed,
                            max_release_speed,
                        ),
                        turn=current.turn[current_indices],
                        y_val=np.clip(
                            current.y_val[current_indices]
                            + self.rng.normal(
                                0,
                                noise_scale * y_width,
                                num_refined_throws,
                            ),
                            min_release_y,
                            max_release_y,
                        ),
                        team=np.full(num_refined_throws, team, dtype=int),
                    )
                )
            current = Throws(
                angle_deg=np.concatenate([t.angle_deg for t in refined])
                .reshape(num_sims, -1)
                .T.ravel(),
                speed=np.concatenate([t.speed for t in refined])
                .reshape(num_sims, -1)
                .T.ravel(),
                turn=np.concatenate([t.turn for t in refined])
                .reshape(num_sims, -1)
                .T.ravel(),
                y_val=np.concatenate([t.y_val for t in refined])
                .reshape(num_sims, -1)
                .T.ravel(),
                team=np.full(num_sims * num_refined_throws, team, dtype=int),
            )

        return current, tile_sheet_states(sheet_states, current.angle_deg.size // num_sims)


class RepulsiveHierarchicalRandomThrows(ThrowSearcher):
    """Hierarchical random search with score-aware diversity selection.

    ``level_configs`` contains ``(num_generated, num_selected)`` for each
    later stage. Each stage generates a fresh random pool without running
    physics, and selects the least-penalized candidates. Physics is then run
    only on those selected candidates. The adjustment is the logarithm of the
    geometric mean of
    ``exp(-distance * c1) * exp(-badness * c2)`` over existing throws.

    Distances are normalized by the configured parameter ranges. Throws are
    partitioned into a 4 x 4 x 4 grid over angle, speed, and release-y, and
    only throws in the same cell are compared. Empty cells receive no
    penalty, allowing one throw from such a cell to be selected and added to
    the existing set for the next level. ``badness`` is the existing throw's
    score deficit from the best existing score, which makes lower-scoring
    existing throws repel more strongly.
    """

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
        existing_scores = score_throws_by_net_score(existing_states, existing).reshape(
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
                        -self.c1 * distance
                        - self.c2 * badness[None, :],
                        axis=1,
                    )
                selected = np.argpartition(log_penalty, -num_selected)[-num_selected:]
                selected_by_state.append(pool_indices[selected])

            selected_indices = np.concatenate(selected_by_state)
            # selected_by_state is state-major; convert to candidate-major.
            selected_indices = selected_indices.reshape(num_sims, num_selected).T.ravel()
            existing = Throws(
                angle_deg=pool.angle_deg[selected_indices],
                speed=pool.speed[selected_indices],
                turn=pool.turn[selected_indices],
                y_val=pool.y_val[selected_indices],
                team=np.full(num_sims * num_selected, team, dtype=int),
            )
            selected_states = tile_sheet_states(sheet_states, num_selected)
            existing_scores = score_throws_by_net_score(
                selected_states, existing
            ).reshape(num_selected, num_sims)

        return existing, tile_sheet_states(sheet_states, existing.angle_deg.size // num_sims)

    @staticmethod
    def _cell_ids(throws: Throws, num_throws: int, num_sims: int) -> np.ndarray:
        """Return 4x4x4 continuous-parameter cell IDs."""
        angle = np.clip(
            ((throws.angle_deg - min_release_angle)
             / (max_release_angle - min_release_angle) * 4).astype(int), 0, 3
        ).reshape(num_throws, num_sims)
        speed = np.clip(
            ((throws.speed - min_release_speed)
             / (max_release_speed - min_release_speed) * 4).astype(int), 0, 3
        ).reshape(num_throws, num_sims)
        y_val = np.clip(
            ((throws.y_val - min_release_y)
             / (max_release_y - min_release_y) * 4).astype(int), 0, 3
        ).reshape(num_throws, num_sims)
        return angle * 16 + speed * 4 + y_val


def score_throws_by_net_score(sheet_states: SheetStates, throws: Throws) -> np.ndarray:
    final_states = physics.run_until_stopping(
        sheet_states=add_stones_from_throws(sheet_states, throws)
    )
    return scoring.get_net_score_for_team(final_states, int(throws.team[0]))


class ArgmaxThrowPolicy(CurlingPolicy):
    def __init__(
        self,
        random_action_prob: float,
        throw_searcher: ThrowSearcher,
        scoring_function: Callable[[SheetStates, Throws], np.ndarray],
    ):
        self.scoring_function = scoring_function
        self.random_action_prob = random_action_prob
        self.throw_searcher = throw_searcher

    @classmethod
    def max_single_turn_score(cls, random_action_prob: float, throw_searcher: ThrowSearcher):
        return cls(
            random_action_prob=random_action_prob,
            scoring_function=score_throws_by_net_score,
            throw_searcher=throw_searcher,
        )

    @classmethod
    def from_q_network(
        cls,
        random_action_prob: float,
        neural_network: curling_nn.QNetwork,
        normalizer: Normalizer,
        throw_searcher: ThrowSearcher,
    ):
        def scoring_function(sheet_states: SheetStates, throws: Throws) -> np.ndarray:
            input_features = curling_nn.QInputFeatures.create_of_sheet_states(
                sheet_states, throws, normalizer
            )
            nn_output = neural_network.run(input_features[:, :, None])
            expected = neural_network.expected_score(nn_output)
            # Network predicts team-0 net score; flip for team 1.
            return np.where(throws.team == 0, 1, -1) * expected

        return cls(
            random_action_prob=random_action_prob,
            scoring_function=scoring_function,
            throw_searcher=throw_searcher,
        )

    @classmethod
    def from_value_network(
        cls,
        random_action_prob: float,
        neural_network: curling_nn.ValueNetwork,
        normalizer: Normalizer,
        throw_searcher: ThrowSearcher,
    ):
        def scoring_function(sheet_states: SheetStates, throws: Throws) -> np.ndarray:
            final_states = physics.run_until_stopping(
                sheet_states=add_stones_from_throws(sheet_states, throws)
            )
            input_features = curling_nn.VInputFeatures.create_of_sheet_states(
                final_states, normalizer
            )
            nn_output = neural_network.run(input_features[:, :, None])
            expected = neural_network.expected_score(nn_output)
            return np.where(throws.team == 0, 1, -1) * expected

        return cls(
            random_action_prob=random_action_prob,
            scoring_function=scoring_function,
            throw_searcher=throw_searcher,
        )

    def make_throws(
        self, sheet_states: SheetStates, team: int, rng: np.random.Generator
    ):
        num_sims = sheet_states.x.shape[0]
        repeated_throws, tiled_starting_states = self.throw_searcher.get_throws_for_num_sims(
            team=team, sheet_states=sheet_states
        )
        scores = self.scoring_function(tiled_starting_states, repeated_throws)
        chosen_throws = scores.reshape((tiled_starting_states.x.shape[0] // num_sims, num_sims)).argmax(axis=0) * num_sims + np.arange(num_sims)
        return Throws(
            angle_deg=repeated_throws.angle_deg[chosen_throws],
            speed=repeated_throws.speed[chosen_throws],
            turn=repeated_throws.turn[chosen_throws],
            y_val=repeated_throws.y_val[chosen_throws],
            team=repeated_throws.team[chosen_throws],
        )


def run_games(
    *,
    num_sims: int,
    num_stones_per_side: int,
    first_player: CurlingPolicy,
    second_player: CurlingPolicy,
    seed: int,
) -> list[SheetStates]:
    current_state = empty_board(num_sims)
    states = [current_state]
    rng = np.random.default_rng(seed=seed)
    for _ in range(num_stones_per_side):
        for team, player in enumerate([first_player, second_player]):
            throws = player.make_throws(states[-1], team, rng)
            current_state = physics.run_until_stopping(
                sheet_states=add_stones_from_throws(current_state, throws)
            )
            states.append(current_state)
    return states
