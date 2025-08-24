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
        force_render_fps=max_fps if not headless else 30,  # clamp FPS
        show_crosswalk=False,
        show_sidewalk=False,
        shadow_range=0 if headless else 1000,  # disable shadows in headless
        multi_thread_render=True
    )


def make_env(rank, headless=True, seed=0):
    """
    Utility for creating parallel MetaDrive environments.
    """
    def _init():
        cfg = get_base_config(headless=headless)
        cfg["start_seed"] = seed + rank  # different seed per env
        env = MetaDriveEnv(cfg)
        return env
    return _init


def train(total_timesteps=100_000_000, headless=True,
          checkpoint_interval=1_000_000, n_envs=4):
    """
    Long-running training loop with periodic checkpointing.
    n_envs=4 to reduce CPU heat.
    """
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "ppo_metadrive_latest.zip")
    final_path = os.path.join(save_dir, "ppo_metadrive_final.zip")

    # use fewer envs to reduce load
    env = SubprocVecEnv([make_env(i, headless=headless) for i in range(n_envs)])

    print(f"🆕 Starting PPO training on CPU with {n_envs} parallel envs (laptop-safe)")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,             # smaller rollout length per env (less memory)
        batch_size=128,           # smaller batch to reduce RAM + CPU load
        learning_rate=3e-4,
        gamma=0.99,
        n_epochs=10,
        clip_range=0.2,
        verbose=1,
        device="cpu",             # use CPU (GPU unused in MetaDrive sim)
        policy_kwargs=dict(
            net_arch=[256, 256, 128]
        ),
    )

    steps_done = 0

    while steps_done < total_timesteps:
        next_checkpoint = min(checkpoint_interval, total_timesteps - steps_done)
        model.learn(total_timesteps=next_checkpoint, reset_num_timesteps=False)

        steps_done = model.num_timesteps
        model.save(model_path)
        print(f"💾 Latest checkpoint saved at {model_path} ({steps_done} steps)")

    model.save(final_path)
    print(f"✅ Final model saved at {final_path}")
    env.close()
