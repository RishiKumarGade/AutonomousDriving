import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from metadrive import MetaDriveEnv


def get_base_config(headless=True, decision_repeat=5, max_fps=30):
    return dict(
        use_render=not headless,
        manual_control=False,
        traffic_density=0.1,
        map=7,
        num_scenarios=1000,
        start_seed=0,
        decision_repeat=decision_repeat,
        force_render_fps=max_fps if not headless else -1,
        show_crosswalk=False,
        show_sidewalk=False,
        shadow_range=0 if headless else 1000,
        multi_thread_render=True
    )


def make_env(rank, headless=True, seed=0):
    def _init():
        cfg = get_base_config(headless=headless)
        cfg["start_seed"] = seed + rank
        return MetaDriveEnv(cfg)
    return _init


def train(total_timesteps=100_000_000, headless=True,
          checkpoint_interval=1_000_000, n_envs=4):

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    latest_path = os.path.join(save_dir, "ppo_metadrive_latest.zip")
    final_path = os.path.join(save_dir, "ppo_metadrive_final.zip")

    env = SubprocVecEnv([make_env(i, headless=headless) for i in range(n_envs)])

    if os.path.exists(latest_path):
        print(f"🔄 Resuming training from {latest_path}")
        model = PPO.load(latest_path, env=env, device="auto")
    else:
        print(f"🆕 Starting PPO training ({n_envs} envs)")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            n_epochs=10,
            clip_range=0.2,
            verbose=1,
            device="auto",
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256, 128]),
            ),
        )

    steps_done = model.num_timesteps

    while steps_done < total_timesteps:
        step_chunk = min(checkpoint_interval, total_timesteps - steps_done)

        model.learn(
            total_timesteps=step_chunk,
            reset_num_timesteps=False
        )

        steps_done = model.num_timesteps
        model.save(latest_path)

        print(f"💾 Checkpoint saved ({steps_done} steps)")

    model.save(final_path)
    print(f"✅ Final model saved at {final_path}")

    env.close()