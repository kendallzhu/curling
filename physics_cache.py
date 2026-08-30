"""Persistent whole-batch cache around the vectorized physics backend."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import physics
import state


def _batch_key(sheet: state.SheetStates, throw: state.Throws) -> str:
    digest = hashlib.sha256()
    for values in (
        sheet.first_team, sheet.x, sheet.y, sheet.velocities.v,
        sheet.velocities.theta, sheet.rotation_directions,
        throw.angle_deg, throw.speed, throw.turn, throw.y_val, throw.team,
    ):
        array = np.asarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class PhysicsCache:
    """Persistent whole-batch cache for vectorized physics calls."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = self.physics_simulations = 0

    def clear_stats(self) -> None:
        self.hits = self.misses = self.physics_simulations = 0

    def stats(self) -> dict[str, float | int]:
        queries = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "physics_simulations": self.physics_simulations,
            "hit_rate": self.hits / queries if queries else 0.0,
        }

    def _path_for(self, key: str) -> Path:
        return self.path / f"{key}.npz"

    def get(self, key: str):
        cache_path = self._path_for(key)
        if not cache_path.exists():
            return None
        with np.load(cache_path) as values:
            return state.SheetStates(
                first_team=np.array(values["first_team"], copy=True),
                x=np.array(values["x"], copy=True),
                y=np.array(values["y"], copy=True),
                velocities=state.Velocities(
                    v=np.array(values["velocity_v"], copy=True),
                    theta=np.array(values["velocity_theta"], copy=True),
                ),
                rotation_directions=np.array(values["rotation_directions"], copy=True),
            )

    def put(self, key: str, value: state.SheetStates) -> None:
        destination = self._path_for(key)
        temporary = self.path / f".{key}.tmp.npz"
        np.savez_compressed(
            temporary,
            first_team=value.first_team,
            x=value.x,
            y=value.y,
            velocity_v=value.velocities.v,
            velocity_theta=value.velocities.theta,
            rotation_directions=value.rotation_directions,
        )
        temporary.replace(destination)


GLOBAL_PHYSICS_CACHE = PhysicsCache(".sequential_training_physics_cache")


class CachedPhysics:
    """Physics facade with the same vectorized interface plus cache lookup."""

    def __init__(self, backend=physics, cache: PhysicsCache = GLOBAL_PHYSICS_CACHE):
        self.backend = backend
        self.cache = cache

    def run_until_stopping(
        self, *, sheet_states: state.SheetStates, throws: state.Throws | None = None
    ) -> state.SheetStates:
        if throws is None:
            base_states = sheet_states
            throws = state.Throws(
                angle_deg=np.zeros(len(sheet_states.x)),
                speed=np.zeros(len(sheet_states.x)),
                turn=np.zeros(len(sheet_states.x), dtype=int),
                y_val=np.zeros(len(sheet_states.x)),
                team=np.zeros(len(sheet_states.x), dtype=int),
            )
        else:
            base_states = sheet_states
            sheet_states = state.add_stones_from_throws(sheet_states, throws)

        count = sheet_states.x.shape[0]
        if count == 0:
            return self.backend.run_until_stopping(sheet_states=sheet_states)
        key = _batch_key(base_states, throws)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.hits += 1
            return cached
        self.cache.misses += 1
        result = self.backend.run_until_stopping(sheet_states=sheet_states)
        self.cache.physics_simulations += count
        self.cache.put(key, result)
        return result

    def run_to_next_collision_or_stop(self, *, sheet_states: state.SheetStates):
        return self.backend.run_to_next_collision_or_stop(sheet_states=sheet_states)


cached_physics = CachedPhysics()
