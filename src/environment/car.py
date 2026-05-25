import math
from dataclasses import dataclass, field


@dataclass
class CarState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # radians, 0 = east, CCW positive
    speed: float = 0.0


class Car:
    MAX_SPEED = 12.0     # m/s
    MAX_STEER = 1.2      # rad/s
    ACCELERATION = 6.0   # m/s^2
    FRICTION = 1.8       # s^-1
    LENGTH = 2.5         # m (for 3D rendering reference)
    WIDTH = 1.2          # m

    @classmethod
    def step(cls, state: CarState, steering: float, throttle: float, dt: float) -> CarState:
        """Pure function kinematic update. Returns new CarState."""
        new_heading = state.heading + steering * cls.MAX_STEER * dt
        new_speed = state.speed + throttle * cls.ACCELERATION * dt - cls.FRICTION * state.speed * dt
        new_speed = max(0.0, min(cls.MAX_SPEED, new_speed))
        new_x = state.x + new_speed * math.cos(new_heading) * dt
        new_y = state.y + new_speed * math.sin(new_heading) * dt
        return CarState(x=new_x, y=new_y, heading=new_heading, speed=new_speed)
