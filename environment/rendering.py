import random
import time

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    CardMaker,
    AmbientLight,
    DirectionalLight,
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

ROAD_HALF_WIDTH = 2.0
ROAD_LENGTH = 70.0
GROUND_SIZE = 220.0

DIRECTIONS = ["N", "S", "E", "W"]

# Mirrors TrafficSignalEnv.phase_directions in custom_env.py.
PHASE_DIRECTIONS = {
    0: ["N", "S"],   # NS-through
    1: ["E", "W"],   # EW-through
    2: ["N", "S"],   # NS-left
    3: ["E", "W"],   # EW-left
    4: [],           # all-red
}

RED = (0.8, 0.05, 0.05, 1)
GREEN = (0.05, 0.75, 0.15, 1)

# Cars are built at this base size, then scaled up uniformly by CAR_SCALE so
# body/cabin/wheel proportions established below stay correct at any size.
CAR_SCALE = 1.35

QUEUE_START_OFFSET = ROAD_HALF_WIDTH + 4.0
QUEUE_SPACING = 3.6
MAX_RENDERED_QUEUE = 7

CAR_COLORS = [
    (0.9, 0.05, 0.05, 1),
    (0.05, 0.35, 0.95, 1),
    (0.98, 0.82, 0.05, 1),
    (0.95, 0.95, 0.95, 1),
    (0.05, 0.65, 0.25, 1),
    (0.55, 0.15, 0.75, 1),
]

# Degrees so each car faces its direction of travel.
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

        self.vehicle_nodes = {d: [] for d in DIRECTIONS}

    def _setup_camera(self):
        self.camLens.setFov(50)
        self.camLens.setNearFar(1.0, 500.0)
        self.camera.setPos(0, -74, 50)
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
        stalk = ROAD_HALF_WIDTH + 0.6
        specs = {
            "N": (1.0, stalk),
            "S": (-1.0, -stalk),
            "E": (stalk, -1.0),
            "W": (-stalk, 1.0),
        }
        self.signal_lights = {}
        for d, (x, y) in specs.items():
            self.make_box(self.render, 0.15, 0.15, 3.0, (0.25, 0.25, 0.25, 1), pos=(x, y, 1.5))
            light = self.make_box(self.render, 0.5, 0.5, 0.5, RED, pos=(x, y, 3.2))
            self.signal_lights[d] = light

    def set_signal_state(self, action):
        green_dirs = PHASE_DIRECTIONS.get(int(action), [])
        for d, node in self.signal_lights.items():
            node.setColor(*(GREEN if d in green_dirs else RED))

    def _queue_position(self, direction, index):
        offset = QUEUE_START_OFFSET + index * QUEUE_SPACING
        if direction == "N":
            return (-1.0, offset, 0.35)
        if direction == "S":
            return (1.0, -offset, 0.35)
        if direction == "E":
            return (offset, -1.0, 0.35)
        return (-offset, 1.0, 0.35)

    def _make_car(self, direction, index):
        color = CAR_COLORS[index % len(CAR_COLORS)]
        car = self.render.attachNewNode(f"car_{direction}_{index}")

        # Full-length lower chassis (hood + trunk).
        self.make_box(car, 1.1, 2.3, 0.5, color, pos=(0, 0, 0.28))

        # Cabin is shorter than the chassis and offset toward the front, so
        # hood/trunk stay exposed and the silhouette reads as a car rather
        # than a stacked two-tier block.
        self.make_box(car, 0.85, 1.15, 0.5, color, pos=(0, -0.15, 0.78))

        # Windshield: dark box pitched forward to suggest sloped glass.
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

    def update_vehicles(self, queue_lengths):
        for d in DIRECTIONS:
            desired = min(int(queue_lengths[d]), MAX_RENDERED_QUEUE)
            nodes = self.vehicle_nodes[d]
            while len(nodes) < desired:
                nodes.append(self._make_car(d, len(nodes)))
            while len(nodes) > desired:
                nodes.pop().removeNode()
            for i, node in enumerate(nodes):
                node.setPos(*self._queue_position(d, i))

    def sync(self, observation, action):
        queue_lengths = {d: observation[i] for i, d in enumerate(DIRECTIONS)}
        self.update_vehicles(queue_lengths)
        self.set_signal_state(action)
        self.graphicsEngine.renderFrame()

    def run_episode(self, env, policy=None, max_steps=500, fps=10, screenshot_dir=None, screenshot_every=1):
        import os

        obs, _info = env.reset()
        self.sync(obs, 4)

        frame_interval = (1.0 / fps) if (fps and not self.headless) else 0.0
        if screenshot_dir:
            os.makedirs(screenshot_dir, exist_ok=True)

        step_count = 0
        if screenshot_dir:
            self.screenshot(os.path.join(screenshot_dir, f"frame_{step_count:04d}.png"))

        while True:
            action = policy(obs) if policy is not None else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.sync(obs, action)
            step_count += 1

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
