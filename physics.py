"""Changing the robot's physics: fixed scale factors for the robustness sweep, random ones for training."""
import gymnasium as gym
import mujoco
import numpy as np


class Physics:
    """Remembers the nominal masses, inertias and frictions of a MuJoCo env and rescales them on request."""

    def __init__(self, env: gym.Env):
        self.model = env.unwrapped.model
        self.mass = self.model.body_mass.copy()
        self.inertia = self.model.body_inertia.copy()
        self.friction = self.model.geom_friction.copy()
        # mj_setConst uses its mjData as scratch space; give it a spare one so the live state is never touched.
        self.scratch = mujoco.MjData(self.model)

    def scale(self, mass: float = 1.0, friction: float = 1.0):
        # Mass and inertia move together (a uniformly denser body); every contact's friction scales together.
        self.model.body_mass[:] = self.mass * mass
        self.model.body_inertia[:] = self.inertia * mass
        self.model.geom_friction[:] = self.friction * friction
        mujoco.mj_setConst(self.model, self.scratch)


class DomainRandomization(gym.Wrapper):
    """At every reset, draw a mass factor and a friction factor uniformly from the given ranges."""

    def __init__(self, env: gym.Env, mass_range, friction_range, seed: int):
        super().__init__(env)
        self.physics = Physics(env)
        self.mass_range, self.friction_range = mass_range, friction_range
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        k_mass, k_friction = self.rng.uniform(*self.mass_range), self.rng.uniform(*self.friction_range)
        self.physics.scale(k_mass, k_friction)
        info.update(mass_scale=k_mass, friction_scale=k_friction)
        return obs, info


class FixedPhysics(gym.Wrapper):
    """Hold the physics at one fixed setting, re-applied at every reset."""

    def __init__(self, env: gym.Env, mass: float = 1.0, friction: float = 1.0):
        super().__init__(env)
        self.physics = Physics(env)
        self.k = (mass, friction)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.physics.scale(*self.k)
        return obs, info
