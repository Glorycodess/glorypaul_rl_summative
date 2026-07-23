"""Smoke test / demo driver for environment/rendering.py.

Default (headless) run:
    uv run python -m tests.test_rendering

Live interactive window (opens a real Panda3D window, plays a random-policy
episode at ~10 fps so you can watch it):
    uv run python -m tests.test_rendering --live

Video-capture mode (headless, dumps a PNG per step for stitching into a demo
video with ffmpeg):
    uv run python -m tests.test_rendering --capture
"""

import argparse
import os

from environment.custom_env import TrafficSignalEnv
from environment.rendering import TrafficRenderer

RENDER_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "renders")


def check_camera_framing(renderer):
    path = os.path.join(RENDER_DIR, "camera_check.png")
    renderer.screenshot(path)
    assert os.path.exists(path), "camera check screenshot was not written"
    print(f"[ok] camera framing screenshot saved to {path}")


def check_signals_and_vehicles(renderer):
    renderer.update_vehicles({"N": 5, "S": 2, "E": 8, "W": 1})
    for action in (0, 1, 2, 3, 4):
        renderer.set_signal_state(action)
        assert set(renderer.signal_lights.keys()) == {"N", "S", "E", "W"}
    renderer.graphicsEngine.renderFrame()
    path = os.path.join(RENDER_DIR, "signals_vehicles_check.png")
    renderer.screenshot(path)
    print(f"[ok] signals + vehicles screenshot saved to {path}")


def run_smoke_episode(renderer, capture):
    env = TrafficSignalEnv()
    screenshot_dir = os.path.join(RENDER_DIR, "episode") if capture else None
    steps = renderer.run_episode(
        env,
        max_steps=20,
        fps=0,
        screenshot_dir=screenshot_dir,
        screenshot_every=5,
    )
    assert steps > 0
    print(f"[ok] ran {steps} env steps synced with the renderer")
    if capture:
        print(f"[ok] frames written to {screenshot_dir}")


def run_live_demo(fps):
    env = TrafficSignalEnv()
    renderer = TrafficRenderer(headless=False)
    steps = renderer.run_episode(env, max_steps=500, fps=fps)
    print(f"live episode finished after {steps} steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="open an interactive window and play a random-policy episode")
    parser.add_argument("--capture", action="store_true", help="headless run that also dumps per-step screenshots")
    parser.add_argument("--fps", type=int, default=10, help="playback frame rate for --live mode")
    args = parser.parse_args()

    os.makedirs(RENDER_DIR, exist_ok=True)

    if args.live:
        run_live_demo(args.fps)
        return

    renderer = TrafficRenderer(headless=True)
    check_camera_framing(renderer)
    check_signals_and_vehicles(renderer)
    run_smoke_episode(renderer, capture=args.capture)
    print("all rendering checks passed")


if __name__ == "__main__":
    main()
