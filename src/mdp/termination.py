import numpy as np
from src.utils import math_utils


class TerminationChecker:
    def __init__(self, cfg):
        self.z_min = float(cfg["z_range"][0])
        self.z_max = float(cfg["z_range"][1])
        self.roll_min = np.deg2rad(cfg["roll_range"][0])
        self.roll_max = np.deg2rad(cfg["roll_range"][1])
        self.pitch_min = np.deg2rad(cfg["pitch_range"][0])
        self.pitch_max = np.deg2rad(cfg["pitch_range"][1])

    def is_healthy(self, state):
        # state[3:7] is quaternion (w, x, y, z)
        # Convert to euler angles (roll, pitch, yaw)
        w, x, y, z = state[3:7]
        roll, pitch, yaw = math_utils.euler_from_quaternion(w, x, y, z)

        return (
            np.isfinite(state).all()
            and self.z_min <= state[2] <= self.z_max
            and self.roll_min <= roll <= self.roll_max
            and self.pitch_min <= pitch <= self.pitch_max
        )
