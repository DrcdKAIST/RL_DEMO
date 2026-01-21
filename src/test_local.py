import time
import yaml
from pathlib import Path

from stable_baselines3 import PPO
from go1_mujoco_env import Go1MujocoEnv

MODEL_DIR = "models"
LOG_DIR = "logs"

def test():
    model_path = "/home/kyu/Desktop/workspace/RL_DEMO/models/2026-01-21_16-34-00/best_model.zip"
    model_path = Path(model_path)

    cfg_path = Path(f"/home/kyu/Desktop/workspace/RL_DEMO/src/params.yaml")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = Go1MujocoEnv(
        prj_path="/home/kyu/Desktop/workspace/RL_DEMO",
        render_mode="human",
        camera_name="tracking",
        width=1920,
        height=1080,
    )

    model = PPO.load(path=model_path, env=env, verbose=1)

    max_time_step_s = cfg["test"]["max_time_step_s"]
    total_reward = 0
    total_length = 0
    if total_length < int(max_time_step_s * 500):
        obs, _ = env.reset()
        env.render()

        ep_len = 0
        ep_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            # Slow down the rendering
            time.sleep(0.1)

            if terminated or truncated:
                print(f"{ep_len=}  {ep_reward=}")
                break

        total_length += ep_len
        total_reward += ep_reward



if __name__ == "__main__":
    test()
