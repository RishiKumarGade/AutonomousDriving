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
        force_render_fps=max_fps if not headless else 0,
        show_crosswalk=False,
        show_sidewalk=False,
        shadow_range=1000 if not headless else 0,
        multi_thread_render=True
    )


def test(model_path, headless=False, live=True, max_episodes=5):
    cfg = get_base_config(headless=headless)
    env = MetaDriveEnv(cfg)

    print(f"🎮 Loading model from {model_path}")
    model = PPO.load(model_path, env=env, device="cpu")

    for ep in range(max_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if not headless and live:
                env.render(mode="topdown", window=True)

        print(f"🏁 Episode {ep+1} reward: {ep_reward:.2f}")

    env.close()
