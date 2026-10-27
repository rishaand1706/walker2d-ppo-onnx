"""Record the exported ONNX controller walking (nominal physics) as a GIF for the README.

    python render.py        # writes assets/walker.gif

The frames come from the ONNX Runtime controller, not the PyTorch policy, so the clip shows what ships.
Every 5th control step is kept (40 ms of sim per frame), which plays back at real time.
"""
import gymnasium as gym
import imageio.v3 as iio

from config import ENV_ID, EVAL_SEED, ROOT
from controller import OnnxController
from export_onnx import FP32

STRIDE, STEPS = 5, 750


def main():
    env = gym.make(ENV_ID, render_mode="rgb_array", width=360, height=200)
    act = OnnxController(FP32)
    obs, _ = env.reset(seed=EVAL_SEED)
    env.render()
    # Gymnasium's offscreen renderer resets the camera to a free (non-tracking) camera on every frame, so
    # follow the torso by moving the look-at point ourselves: a side view, walking left to right.
    cam = env.unwrapped.mujoco_renderer.viewer.cam
    cam.distance, cam.azimuth, cam.elevation = 3.3, 90.0, -8.0
    torso = env.unwrapped.data.body("torso")
    frames = []
    for t in range(STEPS):
        obs, _, terminated, truncated, _ = env.step(act(obs))
        if t % STRIDE == 0:
            cam.lookat[:] = (torso.xpos[0], 0.0, 1.0)
            frames.append(env.render())
        if terminated or truncated:
            break
    out = ROOT / "assets" / "walker.gif"
    out.parent.mkdir(exist_ok=True)
    iio.imwrite(out, frames, duration=round(env.unwrapped.dt * STRIDE * 1000), loop=0)
    print(f"{out.relative_to(ROOT)}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
