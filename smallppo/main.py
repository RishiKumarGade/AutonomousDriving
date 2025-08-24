import argparse
from train_agent import train
from test_agent import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--headless",default=False, action="store_true")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--resume", type=str, help="Path to resume checkpoint")
    parser.add_argument("--ci", type=int, default=100_000)

    args = parser.parse_args()

    if args.mode == "train":
        train(
            total_timesteps=args.timesteps,
            headless=args.headless,
            resume_model=args.resume,
            checkpoint_interval=args.ci
        )
    elif args.mode == "test":
        if not args.model:
            raise ValueError("Please provide --model path for testing.")
        test(model_path=args.model, headless=args.headless)
