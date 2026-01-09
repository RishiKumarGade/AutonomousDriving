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
        force_render_fps=max_fps if not headless else 30, 
        show_crosswalk=False,
        show_sidewalk=False,
        shadow_range=0 if headless else 1000,  
        multi_thread_render=True
    )


def make_env(rank, headless=True, seed=0):

    def _init():
        cfg = get_base_config(headless=headless)
        cfg["start_seed"] = seed + rank  
        env = MetaDriveEnv(cfg)
        return env
    return _init


def train(total_timesteps=50_000_000, headless=True,
          checkpoint_interval=1_000_000, n_envs=8):
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "ppo_metadrive_latest.zip")
    final_path = os.path.join(save_dir, "ppo_metadrive_final.zip")
    env = SubprocVecEnv([make_env(i, headless=headless) for i in range(n_envs)])
    if os.path.exists(model_path):
        print(f"🔄 Resuming training from {model_path}")
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        print(f"🆕 Starting PPO training with {n_envs} parallel envs (balanced fast mode)")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=2048,           
            batch_size=512,          
            learning_rate=3e-4,      
            gamma=0.99,
            n_epochs=10,            
            clip_range=0.2,
            verbose=1,
            device="cpu",
            policy_kwargs=dict(
                net_arch=[256, 256]   
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


def reward_function(self, vehicle_id):
    vehicle = self.vehicles[vehicle_id]
    reward = 0.0
    reward += vehicle.speed * 0.05
    lane = vehicle.lane
    if lane is not None:
        lateral_dist = abs(vehicle.lateral_dist)
        reward -= lateral_dist * 0.1
    if vehicle.crash_vehicle:
        reward -= 10.0
    if vehicle.crash_object:
        reward -= 5.0
    if vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line:
        reward -= 2.0
    reward -= abs(vehicle.steering) * 0.05
    if vehicle.speed < 0.1:
        reward -= 0.2
    return reward
