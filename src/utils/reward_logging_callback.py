from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.buffer = {}

    def _on_step(self) -> bool:
        infos = self.locals["infos"]

        for info in infos:
            for k, v in info.items():
                if k.startswith(("reward/", "cost/")):
                    self.buffer.setdefault(k, []).append(v)

        return True

    def _on_rollout_end(self):
        for k, v in self.buffer.items():
            self.logger.record(k, np.mean(v))
        self.buffer.clear()
