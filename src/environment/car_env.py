import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .car import Car, CarState
from .track import Track


NUM_RAYS = 16
RAY_SPREAD = math.pi  # -90 to +90 degrees from heading
MAX_STEPS = 2000
DT = 1.0 / 60.0
GOAL_RADIUS = 4.0


class CarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.track = Track()

        # 19-dim: 16 raycasts + speed + angle_to_goal + dist_to_goal
        low = np.zeros(NUM_RAYS + 3, dtype=np.float32)
        high = np.ones(NUM_RAYS + 3, dtype=np.float32)
        low[NUM_RAYS + 1] = -1.0   # angle_to_goal can be negative
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # [steering (-1,1), throttle (0,1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.car_state: CarState = CarState()
        self._step_count: int = 0
        self._prev_progress: float = 0.0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        x, y, heading = self.track.get_start_pose()
        # Small random perturbation for generalization
        x += self._rng.uniform(-1.0, 1.0)
        heading += self._rng.uniform(-0.15, 0.15)

        self.car_state = CarState(x=x, y=y, heading=heading, speed=0.0)
        self._step_count = 0
        self._prev_progress = self.track.progress_along_track(x, y)

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        steering = float(action[0])
        throttle = float(action[1])

        self.car_state = Car.step(self.car_state, steering, throttle, DT)
        self._step_count += 1

        x, y = self.car_state.x, self.car_state.y
        in_bounds = self.track.is_in_bounds(x, y)
        gx, gy = self.track.get_goal_position()
        dist_to_goal = math.sqrt((x - gx) ** 2 + (y - gy) ** 2)
        reached_goal = dist_to_goal < GOAL_RADIUS

        # Progress reward
        new_progress = self.track.progress_along_track(x, y)
        delta = new_progress - self._prev_progress
        if delta < -0.5:
            delta += 1.0
        elif delta > 0.5:
            delta -= 1.0
        self._prev_progress = new_progress

        reward = delta * 100.0
        reward -= 0.01  # time penalty

        terminated = False
        if not in_bounds:
            reward -= 5.0
            terminated = True
        if reached_goal:
            reward += 50.0
            terminated = True

        truncated = self._step_count >= MAX_STEPS

        raycasts = self._cast_rays()
        obs = self._build_obs(raycasts, dist_to_goal)
        info = {
            "speed": self.car_state.speed,
            "progress": new_progress,
            "distance_to_goal": dist_to_goal,
            "raycasts": raycasts,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _cast_rays(self):
        angles = np.linspace(
            self.car_state.heading - RAY_SPREAD / 2,
            self.car_state.heading + RAY_SPREAD / 2,
            NUM_RAYS,
        )
        return [self.track.cast_ray(self.car_state.x, self.car_state.y, a) for a in angles]

    def _get_obs(self):
        raycasts = self._cast_rays()
        gx, gy = self.track.get_goal_position()
        dist = math.sqrt((self.car_state.x - gx) ** 2 + (self.car_state.y - gy) ** 2)
        return self._build_obs(raycasts, dist)

    def _build_obs(self, raycasts, dist_to_goal):
        gx, gy = self.track.get_goal_position()
        angle_to_goal = math.atan2(gy - self.car_state.y, gx - self.car_state.x)
        # Relative angle: how far off-heading is the goal
        rel_angle = angle_to_goal - self.car_state.heading
        # Normalize to [-pi, pi] then to [-1, 1]
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        rel_angle_norm = rel_angle / math.pi

        obs = np.array(
            raycasts
            + [
                self.car_state.speed / Car.MAX_SPEED,
                rel_angle_norm,
                min(dist_to_goal / self.track.max_distance, 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def render(self):
        raise NotImplementedError("Use Simulator3D for visualization.")

    def close(self):
        pass
