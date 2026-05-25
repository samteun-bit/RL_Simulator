import math
import sys
import numpy as np

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    AmbientLight, DirectionalLight,
    Camera, PerspectiveLens, NodePath,
    LineSegs, GeomNode, GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomLines,
    Vec3, Vec4, Point3, LColor,
    CardMaker, TextNode,
    WindowProperties, FrameBufferProperties, GraphicsOutput,
    TransparencyAttrib,
)
from stable_baselines3 import PPO

from src.environment.car_env import CarEnv, DT, NUM_RAYS
from src.environment.track import Track
from src.environment.car import Car


class Simulator3D(ShowBase):
    def __init__(self, model_path: str):
        ShowBase.__init__(self)

        self.disableMouse()
        props = WindowProperties()
        props.setTitle("RL Autonomous Driving Simulator")
        props.setSize(1400, 700)
        self.win.requestProperties(props)

        self.env = CarEnv()
        self.obs, _ = self.env.reset()
        self._load_model(model_path)

        self._setup_lighting()
        self._setup_cameras()
        self._build_track()
        self._build_car()
        self._build_goal_marker()
        self._build_ray_lines()
        self._build_hud()

        self._episode_reward = 0.0
        self._episode_steps = 0
        self._laps = 0

        self.taskMgr.add(self._update_task, "SimulatorUpdate")
        self.accept("escape", sys.exit)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str):
        try:
            self.model = PPO.load(model_path, device="cpu")
            print(f"Model loaded: {model_path}")
        except Exception as e:
            print(f"WARNING: Could not load model ({e}). Running with random actions.")
            self.model = None

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------

    def _setup_lighting(self):
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.4, 0.4, 0.4, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.9, 0.85, 0.75, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)
        self.render.setLight(sun_np)

    # ------------------------------------------------------------------
    # Split-screen cameras
    # ------------------------------------------------------------------

    def _setup_cameras(self):
        # Remove the default camera
        base.cam.node().setActive(False)

        # Left panel: chase camera (0.0 – 0.5 of window width)
        self.left_region = self.win.makeDisplayRegion(0.0, 0.5, 0.0, 1.0)
        self.left_region.setClearColor(Vec4(0.08, 0.08, 0.12, 1))
        self.left_region.setClearColorActive(True)
        self.left_region.setClearDepthActive(True)
        self.left_region.setSort(0)

        chase_node = Camera("chase_cam")
        lens = PerspectiveLens()
        lens.setFov(70)
        lens.setNear(0.2)
        lens.setFar(500)
        chase_node.setLens(lens)
        self.chase_cam = self.render.attachNewNode(chase_node)
        self.left_region.setCamera(self.chase_cam)

        # Right panel: car (driver) camera (0.5 – 1.0)
        self.right_region = self.win.makeDisplayRegion(0.5, 1.0, 0.0, 1.0)
        self.right_region.setClearColor(Vec4(0.05, 0.07, 0.15, 1))
        self.right_region.setClearColorActive(True)
        self.right_region.setClearDepthActive(True)
        self.right_region.setSort(0)

        car_cam_node = Camera("car_cam")
        car_lens = PerspectiveLens()
        car_lens.setFov(90)
        car_lens.setNear(0.1)
        car_lens.setFar(200)
        car_cam_node.setLens(car_lens)
        # Will be reparented to car node after car is built
        self.car_cam_np = self.render.attachNewNode(car_cam_node)
        self.right_region.setCamera(self.car_cam_np)

    # ------------------------------------------------------------------
    # Track geometry
    # ------------------------------------------------------------------

    def _build_track(self):
        track = self.env.track

        # Road surface (flat quads)
        self._build_road_surface(track)

        # Walls
        self._build_walls(track)

        # Ground plane
        cm = CardMaker("ground")
        size = max(track.OUTER_W, track.OUTER_H) + 10
        cm.setFrame(-size, size, -size, size)
        ground = self.render.attachNewNode(cm.generate())
        ground.setP(-90)
        ground.setZ(-0.05)
        ground.setColor(0.15, 0.22, 0.15, 1)

    def _build_road_surface(self, track):
        ow, oh = track.OUTER_W, track.OUTER_H
        iw, ih = track.INNER_W, track.INNER_H

        # Four road strips: bottom, top, left, right
        strips = [
            (-ow, -oh, ow, -ih),   # bottom
            (-ow, ih, ow, oh),     # top
            (-ow, -ih, -iw, ih),   # left
            (iw, -ih, ow, ih),     # right
        ]
        for (x1, y1, x2, y2) in strips:
            cm = CardMaker("road")
            cm.setFrame(x1, x2, y1, y2)
            road = self.render.attachNewNode(cm.generate())
            road.setP(-90)
            road.setColor(0.28, 0.28, 0.28, 1)

    def _build_walls(self, track):
        wall_h = 1.8

        for (ax, ay), (bx, by) in track.get_wall_segments():
            segs = LineSegs()
            segs.setThickness(4.0)
            segs.setColor(0.9, 0.85, 0.2, 1)  # yellow barriers
            # Bottom edge
            segs.moveTo(ax, ay, 0.0)
            segs.drawTo(bx, by, 0.0)
            # Top edge
            segs.moveTo(ax, ay, wall_h)
            segs.drawTo(bx, by, wall_h)
            # Vertical pillars every ~5m along the segment
            seg_len = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
            n_pillars = max(2, int(seg_len / 5))
            for i in range(n_pillars + 1):
                t = i / n_pillars
                px = ax + t * (bx - ax)
                py = ay + t * (by - ay)
                segs.moveTo(px, py, 0.0)
                segs.drawTo(px, py, wall_h)
            self.render.attachNewNode(segs.create())

    # ------------------------------------------------------------------
    # Car model (procedural colored box)
    # ------------------------------------------------------------------

    def _build_car(self):
        self.car_root = self.render.attachNewNode("car_root")

        body = self._make_box(Car.LENGTH, Car.WIDTH, 0.7, (0.15, 0.45, 1.0, 1))
        body.reparentTo(self.car_root)
        body.setZ(0.4)

        roof = self._make_box(Car.LENGTH * 0.55, Car.WIDTH * 0.9, 0.5, (0.1, 0.3, 0.8, 1))
        roof.reparentTo(self.car_root)
        roof.setPos(0, 0, 0.4 + 0.7)

        # Headlights (small yellow boxes at front)
        for side in (-0.35, 0.35):
            light_box = self._make_box(0.15, 0.2, 0.15, (1.0, 1.0, 0.4, 1))
            light_box.reparentTo(self.car_root)
            light_box.setPos(Car.LENGTH / 2, side, 0.5)

        # Mount car camera at front of car
        self.car_cam_np.reparentTo(self.car_root)
        self.car_cam_np.setPos(Car.LENGTH / 2 - 0.2, 0, 1.1)
        self.car_cam_np.setHpr(0, -5, 0)  # slight downward tilt

    def _make_box(self, lx, ly, lz, color):
        """Build a simple RGBA colored box from CardMaker faces."""
        root = self.render.attachNewNode("box")
        hx, hy, hz = lx / 2, ly / 2, lz / 2
        faces = [
            # (offset, hpr)
            ((0, hy, hz), (0, 90, 0), lx, lz),    # front (+Y)
            ((0, -hy, hz), (180, 90, 0), lx, lz),  # back (-Y)
            ((hx, 0, hz), (90, 90, 0), ly, lz),    # right (+X)
            ((-hx, 0, hz), (270, 90, 0), ly, lz),  # left (-X)
            ((0, 0, lz), (0, 0, 0), lx, ly),        # top
            ((0, 0, 0), (0, 180, 0), lx, ly),       # bottom
        ]
        for (px, py, pz), hpr, fw, fh in faces:
            cm = CardMaker("face")
            cm.setFrame(-fw / 2, fw / 2, -fh / 2, fh / 2)
            face_np = root.attachNewNode(cm.generate())
            face_np.setPos(px, py, pz)
            face_np.setHpr(*hpr)
            face_np.setColor(*color)
        return root

    # ------------------------------------------------------------------
    # Goal marker
    # ------------------------------------------------------------------

    def _build_goal_marker(self):
        gx, gy = self.env.track.get_goal_position()
        segs = LineSegs()
        segs.setThickness(3.0)
        segs.setColor(0.1, 1.0, 0.2, 1)
        r = 4.0
        steps = 32
        for i in range(steps + 1):
            a = 2 * math.pi * i / steps
            x = gx + r * math.cos(a)
            y = gy + r * math.sin(a)
            if i == 0:
                segs.moveTo(x, y, 0.1)
            else:
                segs.drawTo(x, y, 0.1)
        # Vertical post
        segs.moveTo(gx, gy, 0)
        segs.drawTo(gx, gy, 4.0)
        # Flag at top
        segs.moveTo(gx, gy, 4.0)
        segs.drawTo(gx + 2.0, gy, 4.0)
        segs.drawTo(gx + 2.0, gy, 3.0)
        segs.drawTo(gx, gy, 3.0)
        self.render.attachNewNode(segs.create())

    # ------------------------------------------------------------------
    # Raycast visualization lines
    # ------------------------------------------------------------------

    def _build_ray_lines(self):
        self._ray_segs_np = None  # will be rebuilt each frame

    def _update_ray_lines(self, car_state, raycasts):
        if self._ray_segs_np is not None:
            self._ray_segs_np.removeNode()

        segs = LineSegs()
        segs.setThickness(1.5)
        angles = [
            car_state.heading - math.pi / 2 + i * math.pi / (NUM_RAYS - 1)
            for i in range(NUM_RAYS)
        ]
        for dist_norm, angle in zip(raycasts, angles):
            # Color: red = close, green = far
            r = 1.0 - dist_norm
            g = dist_norm
            segs.setColor(r, g, 0.1, 0.8)
            hit_x, hit_y, _ = self.env.track.cast_ray_world(
                car_state.x, car_state.y, angle
            )
            segs.moveTo(car_state.x, car_state.y, 0.3)
            segs.drawTo(hit_x, hit_y, 0.3)
        self._ray_segs_np = self.render.attachNewNode(segs.create())

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _build_hud(self):
        # Left panel labels (aspect2d x range: approx -2 to 0 for left half)
        self.speed_text = OnscreenText(
            text="Speed: 0.0 m/s", pos=(-1.85, 0.88),
            scale=0.07, fg=(1, 1, 0.3, 1), align=TextNode.ALeft,
            mayChange=True,
        )
        self.reward_text = OnscreenText(
            text="Reward: 0.0", pos=(-1.85, 0.78),
            scale=0.07, fg=(0.3, 1, 0.3, 1), align=TextNode.ALeft,
            mayChange=True,
        )
        self.laps_text = OnscreenText(
            text="Laps: 0", pos=(-1.85, 0.68),
            scale=0.07, fg=(1, 0.5, 0.1, 1), align=TextNode.ALeft,
            mayChange=True,
        )
        OnscreenText(
            text="Top View + Raycasts", pos=(-1.0, -0.90),
            scale=0.065, fg=(0.7, 0.7, 0.7, 1),
        )

        # Right panel labels (x range: approx 0 to 2)
        self.cam_label = OnscreenText(
            text="Driver Camera", pos=(1.0, -0.90),
            scale=0.065, fg=(0.7, 0.7, 0.7, 1),
        )
        self.dist_text = OnscreenText(
            text="Goal: 0.0 m", pos=(0.15, 0.88),
            scale=0.07, fg=(0.5, 0.8, 1.0, 1), align=TextNode.ALeft,
            mayChange=True,
        )

        # Center divider hint
        OnscreenText(
            text="|", pos=(0.0, 0.0), scale=2.0,
            fg=(0.5, 0.5, 0.5, 0.3),
        )

    # ------------------------------------------------------------------
    # Chase camera update
    # ------------------------------------------------------------------

    def _update_chase_cam(self, car_state):
        offset = 14.0
        height = 6.0
        cx = car_state.x - offset * math.cos(car_state.heading)
        cy = car_state.y - offset * math.sin(car_state.heading)
        self.chase_cam.setPos(cx, cy, height)
        self.chase_cam.lookAt(Point3(car_state.x, car_state.y, 0.8))

    # ------------------------------------------------------------------
    # Car node sync
    # ------------------------------------------------------------------

    def _sync_car(self, car_state):
        self.car_root.setPos(car_state.x, car_state.y, 0)
        # Convert physics heading (CCW from east) to Panda3D H (CW from north)
        panda_h = -math.degrees(car_state.heading) + 90
        self.car_root.setH(panda_h)

    # ------------------------------------------------------------------
    # Main update task (60 Hz)
    # ------------------------------------------------------------------

    def _update_task(self, task):
        if self.model is not None:
            action, _ = self.model.predict(self.obs, deterministic=True)
        else:
            action = self.env.action_space.sample()

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += reward
        self._episode_steps += 1

        car_state = self.env.car_state

        self._sync_car(car_state)
        self._update_chase_cam(car_state)
        self._update_ray_lines(car_state, info.get("raycasts", []))

        # HUD updates
        self.speed_text.setText(f"Speed: {info['speed']:.1f} m/s")
        self.reward_text.setText(f"Ep.Reward: {self._episode_reward:.1f}")
        self.dist_text.setText(f"Goal: {info['distance_to_goal']:.1f} m")
        self.laps_text.setText(f"Laps: {self._laps}")

        if terminated or truncated:
            print(
                f"Episode ended | steps={self._episode_steps} "
                f"reward={self._episode_reward:.1f} "
                f"reached_goal={'yes' if info['distance_to_goal'] < 4 else 'no'}"
            )
            self._episode_reward = 0.0
            self._episode_steps = 0
            self._laps += 1
            self.obs, _ = self.env.reset()

        return task.cont
