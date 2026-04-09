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


def test(model_path, headless=False, max_episodes=10, deterministic=True):
    """
    Run evaluation episodes with a trained PPO agent.
    - headless=True: disables GUI rendering (fast eval)
    - live=True: shows topdown render if not headless
    - deterministic=True: stable driving policy (no exploration noise)
    """
    cfg = get_base_config(headless=headless)
    env = MetaDriveEnv(cfg)

    print(f"📂 Loading model from {model_path}")
    model = PPO.load(model_path, env=env, device="cpu")

    episode_rewards, episode_steps = [], []

    for ep in range(1, max_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            if not headless:
                env.render(mode="topdown", window=True)

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        print(f"Episode {ep}/{max_episodes} | Steps: {steps} "
              f"| Reward: {total_reward:.2f} | DoneReason: {info.get('crash_vehicle') or info.get('crash_object') or info.get('out_of_road') or 'Success'}")

    env.close()

    avg_r = sum(episode_rewards) / len(episode_rewards)
    avg_s = sum(episode_steps) / len(episode_steps)
    print(f"\n✅ Evaluation finished over {max_episodes} episodes.")
    print(f"   ➡️ Avg Reward: {avg_r:.2f}, Avg Steps: {avg_s:.1f}")
