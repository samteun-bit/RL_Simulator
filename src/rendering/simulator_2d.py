import math
import time
import numpy as np
import pyglet
from pyglet import shapes
from stable_baselines3.common.callbacks import BaseCallback

from src.environment.car_env import CarEnv, NUM_RAYS
from src.environment.car import Car


# ── Colours ───────────────────────────────────────────────────────────────────

BG          = (20,  20,  30)
TRACK_COL   = (55,  55,  55)
GRASS_COL   = (35,  65,  35)
WALL_COL    = (220, 200,  50)
GOAL_COL    = (50,  220,  80)
SIDEBAR_COL = (15,  15,  25)

CAR_COLORS = [
    (220,  60,  60),  # red
    ( 60, 120, 220),  # blue
    ( 60, 200,  80),  # green
    (220, 140,  40),  # orange
    (160,  60, 220),  # purple
    ( 60, 200, 200),  # cyan
    (220, 220,  60),  # yellow
    (220, 100, 160),  # pink
]

SCREEN_W  = 1350
SCREEN_H  = 820
SIDEBAR_W = 300
VIEW_W    = SCREEN_W - SIDEBAR_W  # 1050


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rotated_rect_verts(cx, cy, length, width, heading):
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    hl, hw = length / 2, width / 2
    corners = [(hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]
    return [(cx + x * cos_h - y * sin_h,
             cy + x * sin_h + y * cos_h) for x, y in corners]


# ── Renderer ──────────────────────────────────────────────────────────────────

class Simulator2D:
    """
    Top-down 2D pyglet renderer.

    demo  mode: loaded model predicts actions, drives N cars.
    watch mode: called by WatchCallback during SB3 training;
                car states come from the live DummyVecEnv envs.
    """

    def __init__(self, envs: list, model=None):
        self.envs      = envs
        self.model     = model
        self.n_cars    = len(envs)
        self.should_stop = False

        self.obs       = [None]     * self.n_cars
        self.actions   = [np.zeros(2)] * self.n_cars
        self.ep_reward = [0.0]      * self.n_cars
        self.ep_steps  = [0]        * self.n_cars
        self.ep_count  = [0]        * self.n_cars

        self.window = pyglet.window.Window(
            SCREEN_W, SCREEN_H,
            caption="RL Autonomous Driving Simulator",
        )
        pyglet.gl.glClearColor(*[c / 255 for c in BG], 1.0)

        # Scale: fit track inside the view area
        track  = envs[0].track
        margin = 50
        self.scale = min(
            (VIEW_W  - 2 * margin) / (2 * track.OUTER_W),
            (SCREEN_H - 2 * margin) / (2 * track.OUTER_H),
        )
        self.ox = VIEW_W  // 2
        self.oy = SCREEN_H // 2

        self._static_batch  = pyglet.graphics.Batch()
        self._static_shapes: list = []
        self._dyn_shapes:   list = []
        self._sidebar_labels: list = []

        self._build_static()

        @self.window.event
        def on_draw():
            self.window.clear()
            self._static_batch.draw()
            for s in self._dyn_shapes:
                s.draw()
            for lbl in self._sidebar_labels:
                lbl.draw()

        @self.window.event
        def on_key_press(symbol, _mod):
            if symbol == pyglet.window.key.ESCAPE:
                self.should_stop = True
                pyglet.app.exit()

        @self.window.event
        def on_close():
            self.should_stop = True

    # ── coordinate helper ─────────────────────────────────────────────────────

    def w2s(self, wx, wy):
        return (self.ox + wx * self.scale,
                self.oy + wy * self.scale)

    # ── static geometry (drawn once) ──────────────────────────────────────────

    def _build_static(self):
        track = self.envs[0].track
        ow, oh = track.OUTER_W, track.OUTER_H
        iw, ih = track.INNER_W, track.INNER_H
        b = self._static_batch
        s = self._static_shapes

        # Outer road
        tl = self.w2s(-ow, -oh)
        s.append(shapes.Rectangle(
            tl[0], tl[1],
            int(2 * ow * self.scale), int(2 * oh * self.scale),
            color=TRACK_COL, batch=b,
        ))
        # Inner grass island
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
        # Goal circle
        gx, gy = track.get_goal_position()
        gc = self.w2s(gx, gy)
        r = max(4, int(4.0 * self.scale))
        s.append(shapes.Circle(*gc, r, color=(*GOAL_COL, 180), batch=b))
        # Goal cross
        s.append(shapes.Line(gc[0]-r, gc[1], gc[0]+r, gc[1], thickness=2, color=GOAL_COL, batch=b))
        s.append(shapes.Line(gc[0], gc[1]-r, gc[0], gc[1]+r, thickness=2, color=GOAL_COL, batch=b))
        # Sidebar background + divider
        s.append(shapes.Rectangle(VIEW_W, 0, SIDEBAR_W, SCREEN_H, color=SIDEBAR_COL, batch=b))
        s.append(shapes.Line(VIEW_W, 0, VIEW_W, SCREEN_H, thickness=2, color=(80, 80, 100), batch=b))
        # Title
        s.append(pyglet.text.Label(
            "Car Stats",
            font_name="Arial", font_size=14, weight="bold",
            x=VIEW_W + SIDEBAR_W // 2, y=SCREEN_H - 18,
            color=(200, 200, 230, 255), anchor_x="center", anchor_y="center",
            batch=b,
        ))

    # ── dynamic frame rendering ───────────────────────────────────────────────

    def _clear_dynamic(self):
        for s in self._dyn_shapes:
            s.delete()
        for lbl in self._sidebar_labels:
            lbl.delete()
        self._dyn_shapes    = []
        self._sidebar_labels = []

    def render(self):
        self._clear_dynamic()
        self._draw_raycasts()
        self._draw_cars()
        self._draw_sidebar()

    def _draw_raycasts(self):
        for i, env in enumerate(self.envs):
            cs  = env.car_state
            col = CAR_COLORS[i % len(CAR_COLORS)]
            dim = tuple(max(0, c - 130) for c in col)
            rays = (self.obs[i][:NUM_RAYS] if self.obs[i] is not None
                    else [1.0] * NUM_RAYS)
            ray_angles = np.linspace(
                cs.heading - math.pi / 2,
                cs.heading + math.pi / 2,
                NUM_RAYS,
            )
            for dist_n, angle in zip(rays, ray_angles):
                hx, hy, _ = env.track.cast_ray_world(cs.x, cs.y, angle)
                p1, p2 = self.w2s(cs.x, cs.y), self.w2s(hx, hy)
                alpha = int(80 + 100 * (1 - dist_n))
                self._dyn_shapes.append(
                    shapes.Line(*p1, *p2, thickness=1, color=(*dim, alpha))
                )

    def _draw_cars(self):
        L = Car.LENGTH * self.scale
        W = max(Car.WIDTH  * self.scale, 5)

        for i, env in enumerate(self.envs):
            cs  = env.car_state
            col = CAR_COLORS[i % len(CAR_COLORS)]
            cx, cy = self.w2s(cs.x, cs.y)

            # Body
            body_verts = _rotated_rect_verts(cx, cy, L, W, cs.heading)
            self._dyn_shapes.append(
                shapes.Polygon(*body_verts, color=(*col, 230))
            )
            # Front stripe (white – marks the front of the car)
            cos_h, sin_h = math.cos(cs.heading), math.sin(cs.heading)
            fc = (cx + L * 0.32 * cos_h, cy + L * 0.32 * sin_h)
            front_verts = _rotated_rect_verts(*fc, L * 0.24, W * 0.96, cs.heading)
            self._dyn_shapes.append(
                shapes.Polygon(*front_verts, color=(255, 255, 255, 200))
            )
            # Car number
            self._sidebar_labels.append(pyglet.text.Label(
                str(i + 1),
                font_name="Arial", font_size=9,
                x=int(cx), y=int(cy),
                color=(255, 255, 255, 230),
                anchor_x="center", anchor_y="center",
            ))

    def _draw_sidebar(self):
        row_h   = (SCREEN_H - 50) // self.n_cars
        bar_max = SIDEBAR_W - 24

        for i in range(self.n_cars):
            cs  = self.envs[i].car_state
            act = self.actions[i]
            col = CAR_COLORS[i % len(CAR_COLORS)]

            y0 = SCREEN_H - 50 - i * row_h   # top of this row
            x0 = VIEW_W + 10
            y  = y0

            # Colour indicator + label
            self._dyn_shapes.append(
                shapes.Rectangle(x0, y - 15, 13, 13, color=col)
            )
            self._sidebar_labels.append(pyglet.text.Label(
                f"Car {i + 1}",
                font_name="Arial", font_size=12, weight="bold",
                x=x0 + 17, y=y - 10,
                color=(*col, 255), anchor_y="center",
            ))
            y -= 20

            # Speed
            self._sidebar_labels.append(pyglet.text.Label(
                f"Spd: {cs.speed:5.1f} m/s",
                font_name="Courier New", font_size=10,
                x=x0, y=y - 9,
                color=(180, 180, 210, 255), anchor_y="center",
            ))
            y -= 17

            # Engine bar + value
            throttle = float(act[1]) if act is not None else 0.0
            self._sidebar_labels.append(pyglet.text.Label(
                f"Eng: {throttle * 100:5.1f}%",
                font_name="Courier New", font_size=10,
                x=x0, y=y - 9,
                color=(170, 210, 170, 255), anchor_y="center",
            ))
            y -= 17
            filled = int(throttle * bar_max)
            eng_col = (50, min(255, 100 + int(throttle * 150)), 50)
            self._dyn_shapes.append(shapes.Rectangle(x0, y - 7, filled, 7, color=eng_col))
            self._dyn_shapes.append(shapes.Rectangle(x0 + filled, y - 7, bar_max - filled, 7, color=(45, 45, 58)))
            y -= 13

            # Steering bar + value
            steering = float(act[0]) if act is not None else 0.0
            self._sidebar_labels.append(pyglet.text.Label(
                f"Str: {steering:+.3f}",
                font_name="Courier New", font_size=10,
                x=x0, y=y - 9,
                color=(210, 170, 170, 255), anchor_y="center",
            ))
            y -= 17
            half     = bar_max // 2
            center_x = x0 + half
            self._dyn_shapes.append(shapes.Rectangle(x0, y - 7, bar_max, 7, color=(45, 45, 58)))
            if steering > 0:
                w = int(steering * half)
                self._dyn_shapes.append(shapes.Rectangle(center_x, y - 7, w, 7, color=(210, 110, 50)))
            elif steering < 0:
                w = int(-steering * half)
                self._dyn_shapes.append(shapes.Rectangle(center_x - w, y - 7, w, 7, color=(50, 110, 210)))
            # Centre marker
            self._dyn_shapes.append(
                shapes.Line(center_x, y - 9, center_x, y + 2, thickness=2, color=(180, 180, 180, 200))
            )
            y -= 13

            # Episode info
            self._sidebar_labels.append(pyglet.text.Label(
                f"Ep:{self.ep_count[i]}  Rwd:{self.ep_reward[i]:+.0f}",
                font_name="Courier New", font_size=9,
                x=x0, y=y - 9,
                color=(120, 120, 155, 255), anchor_y="center",
            ))

            # Row divider
            if i < self.n_cars - 1:
                div_y = y0 - row_h + 2
                self._dyn_shapes.append(shapes.Line(
                    VIEW_W + 5, div_y, SCREEN_W - 5, div_y,
                    thickness=1, color=(48, 48, 62, 255),
                ))

    # ── demo mode ─────────────────────────────────────────────────────────────

    def run_demo(self):
        """Load model → drive N cars → render at 60 FPS."""
        if self.model is None:
            print("No model loaded — using random actions.")
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=i)
            self.obs[i] = obs

        def update(_dt):
            for i, env in enumerate(self.envs):
                act = (self.model.predict(self.obs[i], deterministic=True)[0]
                       if self.model else env.action_space.sample())
                self.actions[i] = act
                obs, rew, term, trunc, _ = env.step(act)
                self.obs[i]        = obs
                self.ep_reward[i] += rew
                self.ep_steps[i]  += 1
                if term or trunc:
                    self.ep_count[i]  += 1
                    self.ep_reward[i]  = 0.0
                    self.ep_steps[i]   = 0
                    self.obs[i], _     = env.reset()
            self.render()

        pyglet.clock.schedule_interval(update, 1 / 60)
        pyglet.app.run()


# ── SB3 training callback ──────────────────────────────────────────────────────

class WatchCallback(BaseCallback):
    """
    Plugs into PPO.learn(); renders the DummyVecEnv envs in real-time.
    Pass the raw env list (same objects inside DummyVecEnv) to Simulator2D.
    """

    def __init__(self, renderer: Simulator2D, render_fps: int = 30):
        super().__init__()
        self.renderer   = renderer
        self.frame_dt   = 1.0 / render_fps
        self._last_draw = 0.0

    def _on_step(self) -> bool:
        self.renderer.window.dispatch_events()
        if self.renderer.should_stop:
            return False

        # Sync actions
        acts = self.locals.get("clipped_actions", self.locals.get("actions"))
        if acts is not None:
            for i in range(min(len(acts), self.renderer.n_cars)):
                self.renderer.actions[i] = np.asarray(acts[i])

        # Sync latest observations
        new_obs = self.locals.get("new_obs")
        if new_obs is not None:
            for i in range(min(len(new_obs), self.renderer.n_cars)):
                self.renderer.obs[i] = new_obs[i]

        # Episode stats
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones",   [])
        for i, (rew, done) in enumerate(zip(rewards, dones)):
            if i >= self.renderer.n_cars:
                break
            self.renderer.ep_reward[i] += float(rew)
            self.renderer.ep_steps[i]  += 1
            if done:
                self.renderer.ep_count[i]  += 1
                self.renderer.ep_reward[i]  = 0.0
                self.renderer.ep_steps[i]   = 0

        # Throttle rendering to render_fps
        now = time.monotonic()
        if now - self._last_draw >= self.frame_dt:
            self.renderer.render()
            self.renderer.window.flip()
            self._last_draw = now

        return True
