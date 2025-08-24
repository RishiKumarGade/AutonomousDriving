import os
from stable_baselines3 import PPO
from metadrive import MetaDriveEnv


def get_base_config(headless=True, decision_repeat=5, max_fps=60):
    return dict(
        use_render=not headless,
        manual_control=False,
        traffic_density=0.05,
        map="C",
        num_scenarios=50,
        start_seed=0,
        decision_repeat=decision_repeat,
        force_render_fps=None if headless else max_fps,  
        show_crosswalk=False,
        show_sidewalk=False,
        shadow_range=1000 if not headless else 0,
        multi_thread_render=True
    )


def train(total_timesteps=50_000, headless=False, resume_model=None,
          checkpoint_interval=20_000):
    """
    Train PPO agent in MetaDrive (lite mode).
    - Lower timesteps, smaller map, fewer scenarios for faster turnaround.
    """
    save_dir = "models_lite"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "ppo_metadrive_latest.zip")
    final_path = os.path.join(save_dir, "ppo_metadrive_final.zip")

    cfg = get_base_config(headless=headless)
    env = MetaDriveEnv(cfg)

    
    if resume_model and os.path.exists(resume_model):
        print(f"🔄 Resuming training from {resume_model}")
        model = PPO.load(resume_model, env=env, device="cpu")
    elif os.path.exists(model_path):
        print(f"🔄 Resuming training from {model_path}")
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        print("🆕 Starting fresh PPO training (lite mode)")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=256,           
            batch_size=64,         
            learning_rate=5e-4,    
            gamma=0.99,
            n_epochs=5,            
            clip_range=0.2,
            verbose=1,
            device="cpu",
            policy_kwargs=dict(
                net_arch=[128, 128]   
            ),
        )

    steps_done = model.num_timesteps

    while steps_done < total_timesteps:
        next_checkpoint = min(checkpoint_interval, total_timesteps - steps_done)
        model.learn(total_timesteps=next_checkpoint, reset_num_timesteps=False)
        steps_done = model.num_timesteps
        model.save(model_path)
        print(f"💾 Checkpoint saved at {model_path} ({steps_done} steps)")

    model.save(final_path)
    print(f"✅ Final model saved at {final_path}")
    env.close()
