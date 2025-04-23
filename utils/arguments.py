import argparse
import time
now = time.strftime('%m%d-%H%M%S', time.localtime())

def get_args():
    parser = argparse.ArgumentParser(description="Running time configurations")
    
    parser.add_argument('--model_path', type=str, default="drl_grasp/models/PPO/best_model.zip")
    parser.add_argument('--test_num', type=int, default=100)
    parser.add_argument('--gui', type=bool, default=False)
    parser.add_argument('--models_dir', type=str, default=f"drl_grasp/models/PPO/{now}")
    parser.add_argument('--logs_dir', type=str, default=f"drl_grasp/logs/PPO/{now}")
    parser.add_argument('--checkpoints', type=str, default=f"drl_grasp/checkpoints/PPO/{now}")
    parser.add_argument('--test', type=str, default=f"drl_grasp/logs/test/{now}")
    parser.add_argument('--timesteps', type=int, default=30000)

    args = parser.parse_args()
    kwargs = vars(args)

    return args, kwargs
