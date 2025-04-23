import os
import sys
import time
from loguru import logger

#  Fix path compatibility to ensure proper import from test/ directory
base_dir = os.path.abspath(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd())
sys.path.append(os.path.join(base_dir, ".."))  # Add parent directory to access FR_Gym
sys.path.append(os.path.join(base_dir, "../utils"))  # Add utils directory if arguments.py is there

from stable_baselines3 import PPO
from FR_Gym.Fr5_env import FR5_Env
from utils.arguments import get_args

if __name__ == '__main__':
    args, kwargs = get_args()
    env = FR5_Env(gui=args.gui)
    env.render()

    # Load trained model
    logger.info(f"Loading model from {args.model_path}")
    model = PPO.load(args.model_path)

    test_num = args.test_num
    success_num = 0

    logger.info(f"Starting test: total {test_num} episodes")
    for i in range(test_num):
        obs, _ = env.reset()
        done = False
        score = 0

        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            score += reward
            time.sleep(0.01)

        if info['is_success']:
            success_num += 1
            logger.success(f"[{i}] Success  | Reward: {score:.2f}")
        else:
            logger.warning(f"[{i}] Failed  | Reward: {score:.2f}")

    success_rate = success_num / test_num
    logger.info(f" Test finished. Success Rate: {success_rate * 100:.2f}% ({success_num}/{test_num})")
    env.close()

