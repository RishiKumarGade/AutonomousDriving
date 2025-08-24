from stable_baselines3 import PPO
from metadrive import MetaDriveEnv

def get_base_config(headless=True, decision_repeat=5, max_fps=60):
    return dict(
        use_render=not headless,
        manual_control=False,
        traffic_density=0.1,
        map=2,
        num_scenarios=50,
        start_seed=0,
        decision_repeat=decision_repeat,
        force_render_fps=max_fps if not headless else -1,
        show_crosswalk=False,
        show_sidewalk=False,
        shadow_range=1000 if not headless else 0,
        multi_thread_render=True
    )

def test(model_path, headless=False, live=True, max_episodes=10):
    cfg = get_base_config(headless=headless)
    env = MetaDriveEnv(cfg)

    model = PPO.load(model_path)

    for ep in range(max_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if not headless and live:
                env.render(mode="topdown", window=True)

    env.close()
