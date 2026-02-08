import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from go1_mujoco_env import Go1MujocoEnv
from pathlib import Path

from utils.reward_logging_callback import RewardLoggingCallback

import yaml

def train():
    base_dir = Path(__file__).resolve().parents[1]
    model_dir = base_dir / "models"
    log_dir = base_dir / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = base_dir / "src" / "params.yaml"

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg["policy"]["use_pretrained"]:
        pretrained_model_path = base_dir / "models" / "pretrained" / "best_model.zip"
    else:
        pretrained_model_path = None

    vec_env = make_vec_env(
        Go1MujocoEnv,
        env_kwargs={"prj_path": base_dir.as_posix()},
        n_envs=cfg["n_envs"],
        seed=cfg["seed"],
        vec_env_cls=SubprocVecEnv,
    )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{train_time}"

    model_path = model_dir / run_name
    print(
        f"Training on {cfg['n_envs']} parallel training environments and saving models to '{model_path}'"
    )

    # Evaluate the model every eval_frequency for 5 episodes and save
    # it if it's improved over the previous best model.
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["policy"]["n_steps"] * cfg["log"]["interval"],  # e.g. 100_000
        save_path=model_path,  # directory
        name_prefix="model",  # checkpoint_model_100000_steps.zip
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=model_path,
        log_path=log_dir,
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    reward_logging_callback = RewardLoggingCallback()

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        reward_logging_callback,
    ])

    if pretrained_model_path is not None:
        print(f"Loading model from {pretrained_model_path}")
        model = PPO.load(
            path=pretrained_model_path,
            env=vec_env,
            learning_rate=cfg["policy"]["learning_rate"],
            n_steps=cfg["policy"]["n_steps"],
            batch_size=cfg["policy"]["batch_size"],
            n_epochs=cfg["policy"]["n_epochs"],
            gamma=cfg["policy"]["gamma"],
            gae_lambda=cfg["policy"]["gae_lambda"],
            clip_range=cfg["policy"]["clip_range"],
            normalize_advantage=cfg["policy"]["normalize_advantage"],
            ent_coef=cfg["policy"]["ent_coef"],
            vf_coef=cfg["policy"]["vf_coef"],
            max_grad_norm=cfg["policy"]["max_grad_norm"],
            verbose=1,
            tensorboard_log=log_dir
        )
    else:
        # Default PPO model hyper-parameters give good results
        model = PPO("MlpPolicy",
                    env=vec_env,
                    learning_rate=cfg["policy"]["learning_rate"],
                    n_steps=cfg["policy"]["n_steps"],
                    batch_size=cfg["policy"]["batch_size"],
                    n_epochs=cfg["policy"]["n_epochs"],
                    gamma=cfg["policy"]["gamma"],
                    gae_lambda=cfg["policy"]["gae_lambda"],
                    clip_range=cfg["policy"]["clip_range"],
                    normalize_advantage=cfg["policy"]["normalize_advantage"],
                    ent_coef=cfg["policy"]["ent_coef"],
                    vf_coef=cfg["policy"]["vf_coef"],
                    max_grad_norm=cfg["policy"]["max_grad_norm"],
                    verbose=1,
                    tensorboard_log=log_dir)

    model.learn(
        total_timesteps=cfg["total_timestep"],
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=run_name,
        callback=callbacks,
    )
    # Save final model
    model.save(model_path / "final_model")


if __name__ == "__main__":
    train()
