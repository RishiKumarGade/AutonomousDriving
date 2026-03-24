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

    env = MetaDriveEnv(get_base_config(headless=headless))

    print(f"📂 Loading model: {model_path}")
    model = PPO.load(model_path, env=env, device="auto")

    rewards, steps_list = [], []

    for ep in range(1, max_episodes + 1):
        obs, _ = env.reset()
        done = False

        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            if not headless:
                env.render()

        rewards.append(total_reward)
        steps_list.append(steps)

        reason = (
            "crash_vehicle" if info.get("crash_vehicle") else
            "crash_object" if info.get("crash_object") else
            "out_of_road" if info.get("out_of_road") else
            "success"
        )

        print(f"Episode {ep} | Steps: {steps} | Reward: {total_reward:.2f} | {reason}")

    env.close()

    print("\n✅ Evaluation complete")
    print(f"Avg Reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Avg Steps: {sum(steps_list)/len(steps_list):.1f}")