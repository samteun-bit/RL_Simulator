"""
2D top-down renderer for the RL car simulator.

Performance design: ALL shapes are pre-allocated at init.
Each frame only updates .x / .y / .rotation / .opacity / .text — zero
GPU allocations during the render loop.
"""

import math
import time
import numpy as np
import pyglet
from pyglet import shapes
from stable_baselines3.common.callbacks import BaseCallback

from src.environment.car_env import CarEnv, NUM_RAYS
from src.environment.car import Car
from src.environment.track import Track


# ── Palette ───────────────────────────────────────────────────────────────────
BG        = (18,  20,  28)
TRACK_COL = (52,  52,  52)
GRASS_COL = (32,  62,  32)
WALL_COL  = (210, 190,  45)

CAR_COLORS = [
    (220,  60,  60), ( 60, 120, 220), ( 60, 200,  80), (220, 140,  40),
    (160,  60, 220), ( 60, 200, 200), (220, 220,  60), (220, 100, 160),
    (150, 220, 180), (255, 140,  90), ( 90, 160, 255), (200, 100, 255),
]

SCREEN_W = 1280
SCREEN_H = 800


# ── Renderer ──────────────────────────────────────────────────────────────────

class Simulator2D:
    def __init__(self, envs: list, model=None, speed: int = 1):
        self.envs    = envs
        self.model   = model
        self.speed   = max(1, speed)
        self.n_cars  = len(envs)
        self.should_stop = False

        self.obs       = [None]        * self.n_cars
        self.actions   = [np.zeros(2)] * self.n_cars
        self.ep_count  = [0]           * self.n_cars
        self.ep_reward = [0.0]         * self.n_cars

        self.window = pyglet.window.Window(
            SCREEN_W, SCREEN_H,
            caption="RL Autonomous Driving Simulator",
        )
        pyglet.gl.glClearColor(*[c / 255 for c in BG], 1.0)

        track  = envs[0].track
        margin = 55
        self.scale = min(
            (SCREEN_W - 2 * margin) / (2 * track.OUTER_W),
            (SCREEN_H - 2 * margin) / (2 * track.OUTER_H),
        )
        self.ox = SCREEN_W // 2
        self.oy = SCREEN_H // 2

        # Two batches: static geometry + dynamic (cars / rays / hud)
        self._static_batch = pyglet.graphics.Batch()
        self._dyn_batch    = pyglet.graphics.Batch()

        self._static_shapes: list = []   # kept alive to prevent GC
        self._build_static()
        self._build_dynamic()            # pre-allocate all per-frame shapes

        @self.window.event
        def on_draw():
            self.window.clear()
            self._static_batch.draw()
            self._dyn_batch.draw()

        @self.window.event
        def on_key_press(symbol, _):
            if symbol == pyglet.window.key.ESCAPE:
                self.should_stop = True
                pyglet.app.exit()

        @self.window.event
        def on_close():
            self.should_stop = True

    # ── helpers ───────────────────────────────────────────────────────────────

    def w2s(self, wx, wy):
        return (self.ox + wx * self.scale, self.oy + wy * self.scale)

    # ── static geometry (drawn once, never rebuilt) ───────────────────────────

    def _build_static(self):
        track = self.envs[0].track
        ow, oh = track.OUTER_W, track.OUTER_H
        iw, ih = track.INNER_W, track.INNER_H
        b = self._static_batch
        s = self._static_shapes

        # Road surface
        tl = self.w2s(-ow, -oh)
        s.append(shapes.Rectangle(
            tl[0], tl[1],
            int(2 * ow * self.scale), int(2 * oh * self.scale),
            color=TRACK_COL, batch=b,
        ))
        # Grass island
        tl2 = self.w2s(-iw, -ih)
        s.append(shapes.Rectangle(
            tl2[0], tl2[1],
            int(2 * iw * self.scale), int(2 * ih * self.scale),
            color=GRASS_COL, batch=b,
        ))
        # Walls
        for (ax, ay), (bx, by) in track.get_wall_segments():
            p1, p2 = self.w2s(ax, ay), self.w2s(bx, by)
            s.append(shapes.Line(*p1, *p2, thickness=3, color=WALL_COL, batch=b))

        # Start / Finish line — checkered blocks
        (sx1, sy1), (sx2, sy2) = track.get_start_finish_line()
        p1, p2 = self.w2s(sx1, sy1), self.w2s(sx2, sy2)
        seg_px = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        n_chk  = max(4, int(seg_px / 12))
        dx, dy = (p2[0]-p1[0])/n_chk, (p2[1]-p1[1])/n_chk
        sw     = max(4, int(self.scale * 1.1))
        for k in range(n_chk):
            col = (255, 255, 255) if k % 2 == 0 else (220, 40, 40)
            ex  = p1[0] + dx * (k + 0.5)
            ey  = p1[1] + dy * (k + 0.5)
            bk  = math.sqrt(dx**2 + dy**2)
            s.append(shapes.Rectangle(
                int(ex - sw//2), int(ey - bk//2),
                sw, max(1, int(bk)),
                color=col, batch=b,
            ))
        # S/F text label
        mx, my = (p1[0]+p2[0])/2 + 14, (p1[1]+p2[1])/2
        s.append(pyglet.text.Label(
            "S/F", font_name="Arial", font_size=11, weight="bold",
            x=int(mx), y=int(my),
            color=(255, 255, 255, 200),
            anchor_x="left", anchor_y="center", batch=b,
        ))

    # ── pre-allocate all dynamic shapes (no per-frame allocation) ─────────────

    def _build_dynamic(self):
        b  = self._dyn_batch
        L  = Car.LENGTH * self.scale
        W  = max(Car.WIDTH * self.scale, 5)

        self._car_bodies:  list = []
        self._car_fronts:  list = []   # direction arrow lines
        self._ray_lines:   list = []   # list of lists
        self._car_labels:  list = []

        for i in range(self.n_cars):
            col = CAR_COLORS[i % len(CAR_COLORS)]
            dim = tuple(max(0, c - 130) for c in col)

            # Body rectangle — anchor at centre so rotation works correctly
            body = shapes.Rectangle(0, 0, L, W, color=(*col, 230), batch=b)
            body.anchor_x = L / 2
            body.anchor_y = W / 2
            self._car_bodies.append(body)

            # Front indicator: white line from centre toward front
            front = shapes.Line(0, 0, 1, 1, thickness=3,
                                color=(255, 255, 255, 210), batch=b)
            self._car_fronts.append(front)

            # Raycast lines
            car_rays = []
            for _ in range(NUM_RAYS):
                ray = shapes.Line(0, 0, 1, 1, thickness=1,
                                  color=(*dim, 80), batch=b)
                car_rays.append(ray)
            self._ray_lines.append(car_rays)

            # Car number label
            lbl = pyglet.text.Label(
                str(i + 1), font_name="Arial", font_size=9,
                x=0, y=0,
                color=(255, 255, 255, 220),
                anchor_x="center", anchor_y="center",
                batch=b,
            )
            self._car_labels.append(lbl)

        # HUD — single label, text updated each frame
        self._hud_lbl = pyglet.text.Label(
            "", font_name="Courier New", font_size=12,
            x=14, y=SCREEN_H - 18,
            color=(200, 200, 200, 200),
            anchor_y="center", batch=b,
        )

    # ── per-frame update (zero allocation) ───────────────────────────────────

    def render(self):
        L       = Car.LENGTH * self.scale
        RAY_MAX = self.envs[0].track.RAY_MAX_DIST

        for i, env in enumerate(self.envs):
            cs    = env.car_state
            cx, cy = self.w2s(cs.x, cs.y)
            cos_h  = math.cos(cs.heading)
            sin_h  = math.sin(cs.heading)

            # Body: update position and rotation
            body = self._car_bodies[i]
            body.x        = cx
            body.y        = cy
            body.rotation = -math.degrees(cs.heading)  # pyglet = CW

            # Front arrow
            fl = self._car_fronts[i]
            fl.x  = cx;           fl.y  = cy
            fl.x2 = cx + (L/2 - 1) * cos_h
            fl.y2 = cy + (L/2 - 1) * sin_h

            # Raycasts — compute hit from stored obs (no extra raycast calls)
            rays = (self.obs[i][:NUM_RAYS] if self.obs[i] is not None
                    else [1.0] * NUM_RAYS)
            ray_angles = np.linspace(
                cs.heading - math.pi / 2,
                cs.heading + math.pi / 2,
                NUM_RAYS,
            )
            for k, (dist_n, angle) in enumerate(zip(rays, ray_angles)):
                d   = dist_n * RAY_MAX
                hx  = cs.x + d * math.cos(angle)
                hy  = cs.y + d * math.sin(angle)
                p1  = self.w2s(cs.x, cs.y)
                p2  = self.w2s(hx, hy)
                ray = self._ray_lines[i][k]
                ray.x = p1[0]; ray.y = p1[1]
                ray.x2 = p2[0]; ray.y2 = p2[1]
                ray.opacity = int(50 + 90 * (1 - dist_n))

            # Car number
            lbl   = self._car_labels[i]
            lbl.x = int(cx)
            lbl.y = int(cy)

        # HUD
        total_laps = sum(self.ep_count)
        self._hud_lbl.text = (
            f"Cars: {self.n_cars}   Laps: {total_laps}   Speed: {self.speed}x"
        )

    def draw_frame(self):
        """Used by WatchCallback — explicit draw (on_draw not called by SB3)."""
        self.render()
        self.window.clear()
        self._static_batch.draw()
        self._dyn_batch.draw()
        self.window.flip()

    # ── demo mode ─────────────────────────────────────────────────────────────

    def run_demo(self):
        if self.model is None:
            print("No model — using random actions.")
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=i)
            self.obs[i] = obs

        def update(_dt):
            for _ in range(self.speed):
                for i, env in enumerate(self.envs):
                    act = (self.model.predict(self.obs[i], deterministic=True)[0]
                           if self.model else env.action_space.sample())
                    self.actions[i] = act
                    obs, rew, term, trunc, _ = env.step(act)
                    self.obs[i]        = obs
                    self.ep_reward[i] += rew
                    if term or trunc:
                        self.ep_count[i]  += 1
                        self.ep_reward[i]  = 0.0
                        self.obs[i], _     = env.reset()
            self.render()   # shapes updated in-place; on_draw draws them

        pyglet.clock.schedule_interval(update, 1 / 60)
        pyglet.app.run()


# ── SB3 training callback ─────────────────────────────────────────────────────

class WatchCallback(BaseCallback):
    def __init__(self, renderer: Simulator2D, speed: int = 1, render_fps: int = 30):
        super().__init__()
        self.renderer     = renderer
        self.render_every = max(1, speed)
        self.frame_dt     = 1.0 / render_fps
        self._last_draw   = 0.0
        self._call_count  = 0

    def _on_step(self) -> bool:
        self._call_count += 1
        self.renderer.window.dispatch_events()
        if self.renderer.should_stop:
            return False

        acts = self.locals.get("clipped_actions", self.locals.get("actions"))
        if acts is not None:
            for i in range(min(len(acts), self.renderer.n_cars)):
                self.renderer.actions[i] = np.asarray(acts[i])

        new_obs = self.locals.get("new_obs")
        if new_obs is not None:
            for i in range(min(len(new_obs), self.renderer.n_cars)):
                self.renderer.obs[i] = new_obs[i]

        for i, (rew, done) in enumerate(zip(
            self.locals.get("rewards", []),
            self.locals.get("dones",   []),
        )):
            if i >= self.renderer.n_cars:
                break
            self.renderer.ep_reward[i] += float(rew)
            if done:
                self.renderer.ep_count[i]  += 1
                self.renderer.ep_reward[i]  = 0.0

        if self._call_count % self.render_every != 0:
            return True
        now = time.monotonic()
        if now - self._last_draw >= self.frame_dt:
            self.renderer.draw_frame()
            self._last_draw = now

        return True
