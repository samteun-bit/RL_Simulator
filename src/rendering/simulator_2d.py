import math
import time
import numpy as np
import pyglet
from pyglet import shapes
from stable_baselines3.common.callbacks import BaseCallback

from src.environment.car_env import CarEnv, NUM_RAYS
from src.environment.car import Car


# ── Colours ───────────────────────────────────────────────────────────────────
BG         = (18,  20,  28)
TRACK_COL  = (52,  52,  52)
GRASS_COL  = (32,  62,  32)
WALL_COL   = (210, 190,  45)
START_COL  = (255, 255, 255)   # start/finish line
FINISH_COL = (240,  50,  50)   # checkered accent

CAR_COLORS = [
    (220,  60,  60),  # red
    ( 60, 120, 220),  # blue
    ( 60, 200,  80),  # green
    (220, 140,  40),  # orange
    (160,  60, 220),  # purple
    ( 60, 200, 200),  # cyan
    (220, 220,  60),  # yellow
    (220, 100, 160),  # pink
    (150, 220, 180),  # mint
    (255, 140,  90),  # salmon
    ( 90, 160, 255),  # sky blue
    (200, 100, 255),  # lavender
]

SCREEN_W = 1280
SCREEN_H = 800


# ── Helper ────────────────────────────────────────────────────────────────────

def _rotated_rect_verts(cx, cy, length, width, heading):
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    hl, hw = length / 2, width / 2
    corners = [(hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]
    return [(cx + x * cos_h - y * sin_h,
             cy + x * sin_h + y * cos_h) for x, y in corners]


# ── Renderer ──────────────────────────────────────────────────────────────────

class Simulator2D:
    """
    Full-screen top-down 2D renderer. No stats sidebar.

    demo  – model drives N cars at `speed` steps per visual frame.
    watch – called by WatchCallback inside PPO.learn().
    """

    def __init__(self, envs: list, model=None, speed: int = 1):
        self.envs    = envs
        self.model   = model
        self.speed   = max(1, speed)   # physics steps per visual frame
        self.n_cars  = len(envs)
        self.should_stop = False

        self.obs       = [None]         * self.n_cars
        self.actions   = [np.zeros(2)]  * self.n_cars
        self.ep_count  = [0]            * self.n_cars
        self.ep_reward = [0.0]          * self.n_cars

        self.window = pyglet.window.Window(
            SCREEN_W, SCREEN_H,
            caption="RL Autonomous Driving Simulator",
        )
        pyglet.gl.glClearColor(*[c / 255 for c in BG], 1.0)

        # Scale: fit track in full window
        track  = envs[0].track
        margin = 55
        self.scale = min(
            (SCREEN_W - 2 * margin) / (2 * track.OUTER_W),
            (SCREEN_H - 2 * margin) / (2 * track.OUTER_H),
        )
        self.ox = SCREEN_W // 2
        self.oy = SCREEN_H // 2

        self._static_batch  = pyglet.graphics.Batch()
        self._static_shapes: list = []
        self._dyn_shapes:    list = []
        self._hud_labels:    list = []

        self._build_static()

        @self.window.event
        def on_draw():
            self.window.clear()
            self._static_batch.draw()
            for s in self._dyn_shapes:
                s.draw()
            for lbl in self._hud_labels:
                lbl.draw()

        @self.window.event
        def on_key_press(symbol, _):
            if symbol == pyglet.window.key.ESCAPE:
                self.should_stop = True
                pyglet.app.exit()

        @self.window.event
        def on_close():
            self.should_stop = True

    # ── coords ────────────────────────────────────────────────────────────────

    def w2s(self, wx, wy):
        return (self.ox + wx * self.scale, self.oy + wy * self.scale)

    # ── static geometry ───────────────────────────────────────────────────────

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

        # ── Start / Finish line ───────────────────────────────────────────────
        (sx1, sy1), (sx2, sy2) = track.get_start_finish_line()
        p1, p2 = self.w2s(sx1, sy1), self.w2s(sx2, sy2)
        seg_len = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        n_checks = max(4, int(seg_len / 12))

        # Alternating white / red checkered blocks along the line
        dx = (p2[0] - p1[0]) / n_checks
        dy = (p2[1] - p1[1]) / n_checks
        stripe_w = int(self.scale * 1.2)
        for k in range(n_checks):
            col = START_COL if k % 2 == 0 else FINISH_COL
            ex = p1[0] + dx * (k + 0.5)
            ey = p1[1] + dy * (k + 0.5)
            block_len = math.sqrt(dx**2 + dy**2)
            # Vertical-ish stripe at each checkpoint position
            s.append(shapes.Rectangle(
                int(ex - stripe_w // 2), int(ey - block_len // 2),
                stripe_w, int(block_len),
                color=col, batch=b,
            ))

        # "START / FINISH" label
        mx = (p1[0] + p2[0]) / 2 + 14
        my = (p1[1] + p2[1]) / 2
        s.append(pyglet.text.Label(
            "S/F",
            font_name="Arial", font_size=11, weight="bold",
            x=int(mx), y=int(my),
            color=(255, 255, 255, 200),
            anchor_x="left", anchor_y="center",
            batch=b,
        ))

    # ── dynamic rendering ─────────────────────────────────────────────────────

    def _clear_dynamic(self):
        for s in self._dyn_shapes:
            s.delete()
        for lbl in self._hud_labels:
            lbl.delete()
        self._dyn_shapes  = []
        self._hud_labels  = []

    def render(self):
        self._clear_dynamic()
        self._draw_raycasts()
        self._draw_cars()
        self._draw_hud()

    def draw_frame(self):
        """Explicitly draw everything – used by WatchCallback."""
        self.render()
        self.window.clear()
        self._static_batch.draw()
        for s in self._dyn_shapes:
            s.draw()
        for lbl in self._hud_labels:
            lbl.draw()
        self.window.flip()

    def _draw_raycasts(self):
        for i, env in enumerate(self.envs):
            cs  = env.car_state
            col = CAR_COLORS[i % len(CAR_COLORS)]
            dim = tuple(max(0, c - 130) for c in col)
            rays = (self.obs[i][:NUM_RAYS] if self.obs[i] is not None
                    else [1.0] * NUM_RAYS)
            angles = np.linspace(
                cs.heading - math.pi / 2,
                cs.heading + math.pi / 2,
                NUM_RAYS,
            )
            for dist_n, angle in zip(rays, angles):
                hx, hy, _ = env.track.cast_ray_world(cs.x, cs.y, angle)
                p1, p2 = self.w2s(cs.x, cs.y), self.w2s(hx, hy)
                alpha = int(55 + 90 * (1 - dist_n))
                self._dyn_shapes.append(
                    shapes.Line(*p1, *p2, thickness=1, color=(*dim, alpha))
                )

    def _draw_cars(self):
        L = Car.LENGTH * self.scale
        W = max(Car.WIDTH * self.scale, 5)

        for i, env in enumerate(self.envs):
            cs  = env.car_state
            col = CAR_COLORS[i % len(CAR_COLORS)]
            cx, cy = self.w2s(cs.x, cs.y)

            # Body
            body_v = _rotated_rect_verts(cx, cy, L, W, cs.heading)
            self._dyn_shapes.append(shapes.Polygon(*body_v, color=(*col, 230)))

            # White front stripe (marks front of car)
            cos_h, sin_h = math.cos(cs.heading), math.sin(cs.heading)
            fc = (cx + L * 0.32 * cos_h, cy + L * 0.32 * sin_h)
            front_v = _rotated_rect_verts(*fc, L * 0.24, W * 0.94, cs.heading)
            self._dyn_shapes.append(shapes.Polygon(*front_v, color=(255, 255, 255, 210)))

            # Car number
            self._hud_labels.append(pyglet.text.Label(
                str(i + 1),
                font_name="Arial", font_size=9,
                x=int(cx), y=int(cy),
                color=(255, 255, 255, 220),
                anchor_x="center", anchor_y="center",
            ))

    def _draw_hud(self):
        """Minimal top-left info: cars, laps, speed multiplier."""
        total_laps = sum(self.ep_count)
        self._hud_labels.append(pyglet.text.Label(
            f"Cars: {self.n_cars}   Laps: {total_laps}   Speed: {self.speed}x",
            font_name="Courier New", font_size=12,
            x=14, y=SCREEN_H - 18,
            color=(200, 200, 200, 200),
            anchor_y="center",
        ))

    # ── demo mode ─────────────────────────────────────────────────────────────

    def run_demo(self):
        if self.model is None:
            print("No model — using random actions.")
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=i)
            self.obs[i] = obs

        def update(_dt):
            for _ in range(self.speed):          # speed multiplier
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
            self.render()

        pyglet.clock.schedule_interval(update, 1 / 60)
        pyglet.app.run()


# ── SB3 training callback ─────────────────────────────────────────────────────

class WatchCallback(BaseCallback):
    """Renders training environments during PPO.learn()."""

    def __init__(self, renderer: Simulator2D, speed: int = 1, render_fps: int = 30):
        super().__init__()
        self.renderer   = renderer
        # render every `render_every` steps (higher speed → skip more renders)
        self.render_every = max(1, speed)
        self.frame_dt     = 1.0 / render_fps
        self._last_draw   = 0.0
        self._call_count  = 0

    def _on_step(self) -> bool:
        self._call_count += 1
        self.renderer.window.dispatch_events()
        if self.renderer.should_stop:
            return False

        # Sync actions
        acts = self.locals.get("clipped_actions", self.locals.get("actions"))
        if acts is not None:
            for i in range(min(len(acts), self.renderer.n_cars)):
                self.renderer.actions[i] = np.asarray(acts[i])

        # Sync observations
        new_obs = self.locals.get("new_obs")
        if new_obs is not None:
            for i in range(min(len(new_obs), self.renderer.n_cars)):
                self.renderer.obs[i] = new_obs[i]

        # Episode stats
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

        # Render at throttled rate, skipping frames based on speed
        if self._call_count % self.render_every != 0:
            return True
        now = time.monotonic()
        if now - self._last_draw >= self.frame_dt:
            self.renderer.draw_frame()
            self._last_draw = now

        return True
