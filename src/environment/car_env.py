import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .car import Car, CarState
from .track import Track


NUM_RAYS  = 16
RAY_SPREAD = math.pi      # front hemisphere (-90° to +90°)
MAX_STEPS  = 3000
DT         = 1.0 / 60.0
LAP_BONUS  = 100.0        # reward for completing a full lap


class CarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.track = Track()

        # 19-dim: 16 raycasts | speed | lap_progress | steps_fraction
        low  = np.zeros(NUM_RAYS + 3, dtype=np.float32)
        high = np.ones (NUM_RAYS + 3, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # [steering (−1,1), throttle (0,1)]
        self.action_space = spaces.Box(
            low =np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.car_state: CarState  = CarState()
        self._step_count: int     = 0
        self._prev_progress: float = 0.0
        self._lap_progress: float  = 0.0   # accumulated forward distance [0,1]
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        x, y, heading = self.track.get_start_pose()
        x       += self._rng.uniform(-1.5, 1.5)
        heading += self._rng.uniform(-0.15, 0.15)

        self.car_state      = CarState(x=x, y=y, heading=heading, speed=0.0)
        self._step_count    = 0
        self._prev_progress = self.track.progress_along_track(x, y)
        self._lap_progress  = 0.0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        action   = np.clip(action, self.action_space.low, self.action_space.high)
        steering = float(action[0])
        throttle = float(action[1])

        self.car_state = Car.step(self.car_state, steering, throttle, DT)
        self._step_count += 1

        x, y     = self.car_state.x, self.car_state.y
        in_bounds = self.track.is_in_bounds(x, y)

        # Progress around track
        new_progress = self.track.progress_along_track(x, y)
        delta = new_progress - self._prev_progress
        if delta < -0.5:
            delta += 1.0
        elif delta > 0.5:
            delta -= 1.0
        self._prev_progress = new_progress
        self._lap_progress += max(0.0, delta)   # only forward counts toward lap

        lap_done = self._lap_progress >= 1.0

        # Reward
        reward  = delta * 100.0          # forward progress (negative going backward)
        reward -= 0.01                   # time penalty

        terminated = False
        if not in_bounds:
            reward    -= 5.0
            terminated = True
        if lap_done:
            reward    += LAP_BONUS
            terminated = True

        truncated = self._step_count >= MAX_STEPS

        raycasts = self._cast_rays()
        obs      = self._build_obs(raycasts)
        info     = {
            "speed":        self.car_state.speed,
            "lap_progress": self._lap_progress,
            "raycasts":     raycasts,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _cast_rays(self):
        angles = np.linspace(
            self.car_state.heading - RAY_SPREAD / 2,
            self.car_state.heading + RAY_SPREAD / 2,
            NUM_RAYS,
        )
        return [self.track.cast_ray(self.car_state.x, self.car_state.y, a)
                for a in angles]

    def _get_obs(self):
        return self._build_obs(self._cast_rays())

    def _build_obs(self, raycasts):
        obs = np.array(
            raycasts + [
                self.car_state.speed / Car.MAX_SPEED,
                min(self._lap_progress, 1.0),
                self._step_count / MAX_STEPS,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def render(self):
        raise NotImplementedError("Use Simulator2D for visualisation.")

    def close(self):
        pass
