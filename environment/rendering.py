import time
import random

from direct.gui.DirectGui import DirectFrame
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    CardMaker,
    AmbientLight,
    DirectionalLight,
    TextNode,
    Vec4,
    Filename,
    loadPrcFileData,
    GeomVertexFormat,
    GeomVertexData,
    GeomVertexWriter,
    Geom,
    GeomTriangles,
    GeomNode,
)

ROAD_HALF_WIDTH = 4.2
ROAD_LENGTH = 90.0
GROUND_SIZE = 320.0

LANE_OFFSET = 2.5

DIRECTIONS = ["N", "S", "E", "W"]

PHASE_DIRECTIONS = {
    0: ["N", "S"],
    1: ["E", "W"],
    2: ["N", "S"],
    3: ["E", "W"],
    4: [],
}

RED = (1.0, 0.0, 0.0, 1)
GREEN = (0.0, 0.95, 0.1, 1)

SIGNAL_LIGHT_SIZE = 1.4
SIGNAL_HOUSING_SIZE = SIGNAL_LIGHT_SIZE + 0.4
SIGNAL_POLE_HEIGHT = 6.0

CAR_SCALE = 1.35

MAX_RENDERED_QUEUE = 7
MAX_RENDERED_DEPARTING = 10

STOP_LINE_DIST = ROAD_HALF_WIDTH + 5.5
FOLLOW_GAP = 4.2
SPAWN_DIST = ROAD_LENGTH / 2.0 - 2.0
EXIT_DIST = -(ROAD_LENGTH / 2.0 - 3.0)
MAX_SPEED = 9.0
ACCEL = 6.0
DECEL = 9.0
SPAWN_SPEED = MAX_SPEED * 0.6

CAR_COLORS = [
    (0.9, 0.05, 0.05, 1),
    (0.05, 0.35, 0.95, 1),
    (0.98, 0.82, 0.05, 1),
    (0.95, 0.95, 0.95, 1),
    (0.05, 0.65, 0.25, 1),
    (0.55, 0.15, 0.75, 1),
]

CAR_HEADING = {"N": 0, "S": 180, "E": 90, "W": -90}

SIDEWALK_WIDTH = 1.6
SIDEWALK_COLOR = (0.68, 0.66, 0.62, 1)
CROSSWALK_COLOR = (0.92, 0.92, 0.88, 1)
LANE_COLOR = (0.9, 0.85, 0.3, 1)

BUILDING_COLORS = [
    (0.75, 0.55, 0.35, 1),
    (0.85, 0.8, 0.65, 1),
    (0.55, 0.6, 0.65, 1),
    (0.6, 0.35, 0.3, 1),
    (0.7, 0.7, 0.5, 1),
    (0.5, 0.45, 0.4, 1),
    (0.35, 0.5, 0.45, 1),
]

DEFAULT_LIVE_FPS = 10

PHASE_NAMES = {0: "NS-THROUGH", 1: "EW-THROUGH", 2: "NS-LEFT", 3: "EW-LEFT", 4: "ALL-RED"}

HUD_BG_COLOR = (0.05, 0.05, 0.08, 0.65)
HUD_TEXT_COLOR = (0.95, 0.95, 0.95, 1)
HUD_GREEN = (0.3, 0.9, 0.35, 1)
HUD_RED = (0.95, 0.3, 0.3, 1)
HUD_WARNING_COLOR = (0.95, 0.65, 0.1, 1)

HUD_WARNING_FRACTION = 0.75


def _build_box_geom(sx, sy, sz):
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    fmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData("box", fmt, Geom.UHStatic)
    vdata.setNumRows(24)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")

    faces = [
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        ((0, 1, 0), [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)]),
        ((1, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
        ((-1, 0, 0), [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)]),
    ]

    tris = GeomTriangles(Geom.UHStatic)
    idx = 0
    for n, corners in faces:
        for c in corners:
            vertex.addData3(*c)
            normal.addData3(*n)
        tris.addVertices(idx, idx + 1, idx + 2)
        tris.addVertices(idx, idx + 2, idx + 3)
        idx += 4

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("box")
    node.addGeom(geom)
    return node


class _Vehicle:
    """A persistent, continuously-moving car. `dist` is signed distance from
    the intersection center along its direction's approach axis. `state` is
    "approaching" (still part of the queue, driving up to or waiting at the
    stop line) or "departing" (released, driving through and away)."""

    __slots__ = ("node", "dist", "speed", "state")

    def __init__(self, node, dist, speed, state):
        self.node = node
        self.dist = dist
        self.speed = speed
        self.state = state


class TrafficRenderer(ShowBase):
    def __init__(self, headless=False, window_size=(1280, 720)):
        if headless:
            loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", f"win-size {window_size[0]} {window_size[1]}")
        loadPrcFileData("", "sync-video 0")

        super().__init__()
        self.headless = headless

        self.disableMouse()
        self._setup_camera()
        self._setup_lighting()
        self._build_scene()
        self._build_sidewalks()
        self._build_lane_markings()
        self._build_crosswalks()
        self._build_streetlights()
        self._build_buildings()
        self._build_signals()
        self._build_hud()

        self.vehicles = {d: [] for d in DIRECTIONS}
        self._next_color_index = 0
        self.episode_num = 0
        self.episode_reward = 0.0

    def _setup_camera(self):
        self.camLens.setFov(50)
        self.camLens.setNearFar(1.0, 500.0)
        self.camera.setPos(0, -110, 76)
        self.camera.lookAt(0, 0, 0)

    def _setup_lighting(self):
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.45, 0.45, 0.45, 1))
        ambient_node = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_node)

        directional = DirectionalLight("directional")
        directional.setColor(Vec4(0.85, 0.85, 0.8, 1))
        directional_node = self.render.attachNewNode(directional)
        directional_node.setHpr(45, -60, 0)
        self.render.setLight(directional_node)

    def _build_scene(self):
        ground = self._make_flat_card(size=GROUND_SIZE, color=(0.22, 0.5, 0.22, 1))
        ground.setPos(0, 0, -0.02)

        road_ns = self._make_flat_card(size=2 * ROAD_HALF_WIDTH, color=(0.15, 0.15, 0.15, 1))
        road_ns.setScale(1, 1, ROAD_LENGTH / (2 * ROAD_HALF_WIDTH))
        road_ns.setPos(0, 0, -0.01)

        road_ew = self._make_flat_card(size=2 * ROAD_HALF_WIDTH, color=(0.15, 0.15, 0.15, 1))
        road_ew.setScale(ROAD_LENGTH / (2 * ROAD_HALF_WIDTH), 1, 1)
        road_ew.setPos(0, 0, -0.01)

    def _make_flat_card(self, size, color):
        cm = CardMaker("card")
        cm.setFrame(-size / 2, size / 2, -size / 2, size / 2)
        card = self.render.attachNewNode(cm.generate())
        card.setP(-90)
        card.setColor(*color)
        return card

    def _make_rect(self, cx, cy, sx, sy, z, color):
        cm = CardMaker("rect")
        cm.setFrame(-sx / 2, sx / 2, -sy / 2, sy / 2)
        card = self.render.attachNewNode(cm.generate())
        card.setP(-90)
        card.setColor(*color)
        card.setPos(cx, cy, z)
        return card

    def _build_sidewalks(self):
        arm_outer = ROAD_LENGTH / 2.0
        arm_len = arm_outer - ROAD_HALF_WIDTH
        arm_mid = (ROAD_HALF_WIDTH + arm_outer) / 2.0
        sw_center = ROAD_HALF_WIDTH + SIDEWALK_WIDTH / 2.0

        for sign in (1, -1):
            self._make_rect(sw_center, sign * arm_mid, SIDEWALK_WIDTH, arm_len, 0.01, SIDEWALK_COLOR)
            self._make_rect(-sw_center, sign * arm_mid, SIDEWALK_WIDTH, arm_len, 0.01, SIDEWALK_COLOR)
            self._make_rect(sign * arm_mid, sw_center, arm_len, SIDEWALK_WIDTH, 0.01, SIDEWALK_COLOR)
            self._make_rect(sign * arm_mid, -sw_center, arm_len, SIDEWALK_WIDTH, 0.01, SIDEWALK_COLOR)

    def _build_lane_markings(self):
        dash_len = 1.2
        gap = 1.0
        period = dash_len + gap
        arm_outer = ROAD_LENGTH / 2.0
        start = 5.0
        count = int((arm_outer - start) / period)
        for i in range(count):
            offset = start + i * period + dash_len / 2
            self._make_rect(0.0, offset, 0.25, dash_len, 0.0, LANE_COLOR)
            self._make_rect(0.0, -offset, 0.25, dash_len, 0.0, LANE_COLOR)
            self._make_rect(offset, 0.0, dash_len, 0.25, 0.0, LANE_COLOR)
            self._make_rect(-offset, 0.0, dash_len, 0.25, 0.0, LANE_COLOR)

    def _build_crosswalks(self):
        stripe_w = 0.3
        gap = 0.3
        n_stripes = 4
        span = 2 * ROAD_HALF_WIDTH - 0.4
        start = ROAD_HALF_WIDTH + 0.3
        for i in range(n_stripes):
            offset = start + i * (stripe_w + gap)
            self._make_rect(0.0, offset, span, stripe_w, -0.005, CROSSWALK_COLOR)
            self._make_rect(0.0, -offset, span, stripe_w, -0.005, CROSSWALK_COLOR)
            self._make_rect(offset, 0.0, stripe_w, span, -0.005, CROSSWALK_COLOR)
            self._make_rect(-offset, 0.0, stripe_w, span, -0.005, CROSSWALK_COLOR)

    def _make_streetlight(self, x, y):
        self.make_box(self.render, 0.12, 0.12, 4.2, (0.2, 0.2, 0.2, 1), pos=(x, y, 2.1))
        self.make_box(self.render, 0.5, 0.5, 0.3, (0.95, 0.85, 0.5, 1), pos=(x, y, 4.35))

    def _build_streetlights(self):
        lamp_offset = ROAD_HALF_WIDTH + SIDEWALK_WIDTH + 0.5
        arm_outer = ROAD_LENGTH / 2.0
        for frac in (0.25, 0.65):
            d = ROAD_HALF_WIDTH + frac * (arm_outer - ROAD_HALF_WIDTH)
            positions = [
                (lamp_offset, d), (-lamp_offset, d),
                (lamp_offset, -d), (-lamp_offset, -d),
                (d, lamp_offset), (d, -lamp_offset),
                (-d, lamp_offset), (-d, -lamp_offset),
            ]
            for x, y in positions:
                self._make_streetlight(x, y)

    def _build_buildings(self):
        rng = random.Random(7)
        inner = ROAD_HALF_WIDTH + SIDEWALK_WIDTH + 1.5
        outer = ROAD_LENGTH / 2.0 - 3.0
        cells = 3
        cell_size = (outer - inner) / cells

        for sx in (1, -1):
            for sy in (1, -1):
                for i in range(cells):
                    for j in range(cells):
                        if rng.random() > 0.78:
                            continue
                        cx = inner + (i + 0.5) * cell_size + rng.uniform(-cell_size * 0.15, cell_size * 0.15)
                        cy = inner + (j + 0.5) * cell_size + rng.uniform(-cell_size * 0.15, cell_size * 0.15)
                        w = rng.uniform(2.5, 4.5)
                        d = rng.uniform(2.5, 4.5)
                        h = rng.uniform(3.0, 9.0)
                        color = rng.choice(BUILDING_COLORS)
                        self.make_box(self.render, w, d, h, color, pos=(sx * cx, sy * cy, h / 2))

    def make_box(self, parent, sx, sy, sz, color, pos=(0, 0, 0), hpr=(0, 0, 0)):
        node = _build_box_geom(sx, sy, sz)
        np_ = parent.attachNewNode(node)
        np_.setTwoSided(True)
        np_.setColor(*color)
        np_.setPos(*pos)
        np_.setHpr(*hpr)
        return np_

    def _build_signals(self):
        stalk = ROAD_HALF_WIDTH + 1.5
        median_offset = 0.35
        specs = {
            "N": (-median_offset, stalk),
            "S": (median_offset, -stalk),
            "E": (stalk, -median_offset),
            "W": (-stalk, median_offset),
        }
        housing_z = SIGNAL_POLE_HEIGHT + SIGNAL_HOUSING_SIZE / 2.0
        light_z = SIGNAL_POLE_HEIGHT + SIGNAL_HOUSING_SIZE + SIGNAL_LIGHT_SIZE / 2.0

        self.signal_lights = {}
        for d, (x, y) in specs.items():
            self.make_box(self.render, 0.3, 0.3, SIGNAL_POLE_HEIGHT, (0.15, 0.15, 0.15, 1), pos=(x, y, SIGNAL_POLE_HEIGHT / 2.0))
            self.make_box(self.render, SIGNAL_HOUSING_SIZE, SIGNAL_HOUSING_SIZE, SIGNAL_HOUSING_SIZE, (0.08, 0.08, 0.08, 1), pos=(x, y, housing_z))
            light = self.make_box(self.render, SIGNAL_LIGHT_SIZE, SIGNAL_LIGHT_SIZE, SIGNAL_LIGHT_SIZE, RED, pos=(x, y, light_z))
            self.signal_lights[d] = light

    def set_signal_state(self, action):
        green_dirs = PHASE_DIRECTIONS.get(int(action), [])
        for d, node in self.signal_lights.items():
            node.setColor(*(GREEN if d in green_dirs else RED))

    def _make_2d_rect(self, parent, x, z, width, height, color):
        cm = CardMaker("hud_rect")
        cm.setFrame(0, 1, -height / 2, height / 2)
        node = parent.attachNewNode(cm.generate())
        node.setPos(x, 0, z)
        node.setScale(max(width, 0.001), 1, 1)
        node.setColor(*color)
        return node

    def _build_hud(self):
        panel_left, panel_right = -1.78, -1.00
        panel_top, panel_bottom = 1.0, 0.40

        self.hud_root = self.aspect2d.attachNewNode("hud")

        DirectFrame(
            parent=self.hud_root,
            frameColor=HUD_BG_COLOR,
            frameSize=(panel_left, panel_right, panel_bottom, panel_top),
            pos=(0, 0, 0),
        )

        text_x = panel_left + 0.05
        line_height = 0.05
        y = [panel_top - 0.068]

        def add_line(scale=0.031, fg=HUD_TEXT_COLOR):
            node = OnscreenText(
                text="",
                pos=(text_x, y[0]),
                scale=scale,
                fg=fg,
                align=TextNode.ALeft,
                mayChange=True,
                parent=self.hud_root,
            )
            y[0] -= line_height
            return node

        self.hud_step_text = add_line()
        self.hud_phase_text = add_line()

        y[0] -= 0.014
        bar_x = text_x + 0.21
        self.hud_bar_max_width = panel_right - bar_x - 0.05
        bar_height = 0.025

        self.hud_queue_texts = {}
        self.hud_bar_fills = {}
        for d in DIRECTIONS:
            line_y = y[0]
            self.hud_queue_texts[d] = add_line(scale=0.03)
            self._make_2d_rect(self.hud_root, bar_x, line_y + 0.01, self.hud_bar_max_width, bar_height, (0.3, 0.3, 0.3, 0.8))
            self.hud_bar_fills[d] = self._make_2d_rect(self.hud_root, bar_x, line_y + 0.01, 0.001, bar_height, HUD_GREEN)

        y[0] -= 0.014
        self.hud_step_reward_text = add_line(scale=0.03)
        self.hud_episode_reward_text = add_line(scale=0.03)

        y[0] -= 0.014
        self.hud_status_text = add_line(scale=0.032, fg=HUD_GREEN)

    def _update_hud(self, observation, action, step_count, max_steps, reward, gridlock_threshold):
        queue_lengths = {d: observation[i] for i, d in enumerate(DIRECTIONS)}
        green_dirs = PHASE_DIRECTIONS.get(int(action), [])

        self.hud_step_text.setText(f"Step: {step_count} / {max_steps}    Episode: {self.episode_num}")

        self.hud_phase_text.setText(f"Phase: {PHASE_NAMES.get(int(action), '?')}")
        self.hud_phase_text.setFg(HUD_RED if int(action) == 4 else HUD_GREEN)

        max_queue = 0
        for d in DIRECTIONS:
            q = int(queue_lengths[d])
            max_queue = max(max_queue, q)
            color = HUD_GREEN if d in green_dirs else HUD_RED
            self.hud_queue_texts[d].setText(f"{d}: {q}")
            self.hud_queue_texts[d].setFg(color)
            fraction = min(q / gridlock_threshold, 1.0)
            self.hud_bar_fills[d].setSx(max(self.hud_bar_max_width * fraction, 0.001))
            self.hud_bar_fills[d].setColor(*color)

        self.hud_step_reward_text.setText(f"Step Reward: {reward:.1f}")
        self.hud_episode_reward_text.setText(f"Episode Reward: {self.episode_reward:.1f}")

        if max_queue >= HUD_WARNING_FRACTION * gridlock_threshold:
            self.hud_status_text.setText("STATUS: WARNING - APPROACHING GRIDLOCK")
            self.hud_status_text.setFg(HUD_WARNING_COLOR)
        else:
            self.hud_status_text.setText("STATUS: NORMAL")
            self.hud_status_text.setFg(HUD_GREEN)

    def _position_for(self, direction, dist):
        if direction == "N":
            return (-LANE_OFFSET, dist, 0.35)
        if direction == "S":
            return (LANE_OFFSET, -dist, 0.35)
        if direction == "E":
            return (dist, -LANE_OFFSET, 0.35)
        return (-dist, LANE_OFFSET, 0.35)

    def _make_car(self, direction, color_index):
        color = CAR_COLORS[color_index % len(CAR_COLORS)]
        car = self.render.attachNewNode(f"car_{direction}_{color_index}")

        self.make_box(car, 1.1, 2.3, 0.5, color, pos=(0, 0, 0.28))

        self.make_box(car, 0.85, 1.15, 0.5, color, pos=(0, -0.15, 0.78))

        windshield = self.make_box(
            car, 0.75, 0.08, 0.42, (0.08, 0.08, 0.1, 1), pos=(0, -0.73, 0.63)
        )
        windshield.setP(35)

        wheel_color = (0.05, 0.05, 0.05, 1)
        for wx, wy in [(-0.5, 0.75), (0.5, 0.75), (-0.5, -0.75), (0.5, -0.75)]:
            self.make_box(car, 0.24, 0.42, 0.34, wheel_color, pos=(wx, wy, 0.14))

        headlight_color = (0.95, 0.95, 0.75, 1)
        taillight_color = (0.6, 0.05, 0.05, 1)
        for x in (-0.4, 0.4):
            self.make_box(car, 0.18, 0.05, 0.15, headlight_color, pos=(x, -1.13, 0.3))
            self.make_box(car, 0.18, 0.05, 0.15, taillight_color, pos=(x, 1.13, 0.3))

        car.setH(CAR_HEADING[direction])
        car.setScale(CAR_SCALE)
        return car

    def reset_vehicles(self):
        for d in DIRECTIONS:
            for v in self.vehicles[d]:
                v.node.removeNode()
            self.vehicles[d] = []

    def _spawn_vehicle(self, direction, dist):
        color_index = self._next_color_index
        self._next_color_index += 1
        node = self._make_car(direction, color_index)
        vehicle = _Vehicle(node, dist, SPAWN_SPEED, "approaching")
        node.setPos(*self._position_for(direction, dist))
        self.vehicles[direction].append(vehicle)

    def reconcile_vehicles(self, queue_lengths):
        """Spawns/releases vehicles so the count still "approaching" (i.e.
        not yet released through the intersection) matches the env's true
        queue length for that direction. Only the net per-step delta is
        observable (the env doesn't expose arrivals/clearances separately),
        so a step with simultaneous arrivals and clearances that happen to
        cancel out won't visibly spawn or release anything -- the count
        stays correct, individual vehicle identity is a best-effort
        visualization on top of that.
        """
        for d in DIRECTIONS:
            approaching = [v for v in self.vehicles[d] if v.state == "approaching"]
            delta = int(queue_lengths[d]) - len(approaching)

            if delta > 0:
                room = MAX_RENDERED_QUEUE - len(approaching)
                back_dist = max((v.dist for v in self.vehicles[d]), default=SPAWN_DIST - FOLLOW_GAP)
                for _ in range(max(0, min(delta, room))):
                    spawn_dist = max(SPAWN_DIST, back_dist + FOLLOW_GAP)
                    self._spawn_vehicle(d, spawn_dist)
                    back_dist = spawn_dist
            elif delta < 0:
                approaching.sort(key=lambda v: v.dist)
                departing_count = sum(1 for v in self.vehicles[d] if v.state == "departing")
                for v in approaching[: min(-delta, len(approaching))]:
                    if departing_count >= MAX_RENDERED_DEPARTING:
                        v.node.removeNode()
                        self.vehicles[d].remove(v)
                    else:
                        v.state = "departing"
                        departing_count += 1

    def step_vehicles(self, dt, action):
        served = set(PHASE_DIRECTIONS.get(int(action), []))

        for d in DIRECTIONS:
            direction_served = d in served
            vehicles = self.vehicles[d]
            vehicles.sort(key=lambda v: v.dist)

            kept = []
            ahead_dist = None
            for v in vehicles:
                committed = v.dist < STOP_LINE_DIST
                if committed and v.state == "approaching":
                    v.state = "departing"
                must_hold = not committed and not direction_served

                follow_target = ahead_dist + FOLLOW_GAP if ahead_dist is not None else None
                stop_target = STOP_LINE_DIST if must_hold else None
                if follow_target is not None and stop_target is not None:
                    target = max(follow_target, stop_target)
                else:
                    target = follow_target if follow_target is not None else stop_target

                if target is None:
                    v.speed = min(v.speed + ACCEL * dt, MAX_SPEED)
                else:
                    gap_remaining = v.dist - target
                    braking_distance = (v.speed ** 2) / (2 * DECEL)
                    if gap_remaining <= braking_distance:
                        v.speed = max(v.speed - DECEL * dt, 0.0)
                    else:
                        v.speed = min(v.speed + ACCEL * dt, MAX_SPEED)

                new_dist = v.dist - v.speed * dt
                if target is not None and new_dist < target:
                    new_dist = target
                new_dist = min(new_dist, v.dist)
                if target is not None and new_dist <= target:
                    v.speed = 0.0
                v.dist = new_dist
                v.node.setPos(*self._position_for(d, v.dist))
                ahead_dist = v.dist

                if v.dist < EXIT_DIST:
                    v.node.removeNode()
                else:
                    kept.append(v)

            self.vehicles[d] = kept

    def sync(self, observation, action, dt):
        queue_lengths = {d: observation[i] for i, d in enumerate(DIRECTIONS)}
        self.reconcile_vehicles(queue_lengths)
        self.step_vehicles(dt, action)
        self.set_signal_state(action)
        self.graphicsEngine.renderFrame()

    def run_episode(self, env, policy=None, max_steps=500, fps=DEFAULT_LIVE_FPS, screenshot_dir=None, screenshot_every=1):
        import os

        self.episode_num += 1
        self.episode_reward = 0.0
        self.reset_vehicles()

        dt = 1.0 / (fps if fps else DEFAULT_LIVE_FPS)

        obs, _info = env.reset()
        self._update_hud(obs, 4, 0, max_steps, 0.0, env.gridlock_threshold)
        self.sync(obs, 4, dt)

        frame_interval = (1.0 / fps) if (fps and not self.headless) else 0.0
        if screenshot_dir:
            os.makedirs(screenshot_dir, exist_ok=True)

        step_count = 0
        if screenshot_dir:
            self.screenshot(os.path.join(screenshot_dir, f"frame_{step_count:04d}.png"))

        while True:
            action = policy(obs) if policy is not None else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            self.episode_reward += reward

            self._update_hud(obs, action, step_count, max_steps, reward, env.gridlock_threshold)
            self.sync(obs, action, dt)

            if screenshot_dir and step_count % screenshot_every == 0:
                self.screenshot(os.path.join(screenshot_dir, f"frame_{step_count:04d}.png"))

            if frame_interval:
                time.sleep(frame_interval)

            if terminated or truncated or step_count >= max_steps:
                break

        return step_count

    def screenshot(self, path):
        self.graphicsEngine.renderFrame()
        self.graphicsEngine.renderFrame()
        self.win.saveScreenshot(Filename.fromOsSpecific(path))

    def step(self):
        self.taskMgr.step()
