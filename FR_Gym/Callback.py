from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, CheckpointCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.n_envs = 16  # Number of parallel environments
        
        self.episode_lengths = [0 for _ in range(self.n_envs)]
        self.episode_counts = [0 for _ in range(self.n_envs)]

        self.episode_total_rewards = [0.0 for _ in range(self.n_envs)]
        self.episode_pose_rewards = [0.0 for _ in range(self.n_envs)]
        self.episode_dis_rewards = [0.0 for _ in range(self.n_envs)]
        self.episode_success = [0.0 for _ in range(self.n_envs)]

        self.log_interval = 30  # Logging interval: once every 30 episodes

    def _on_step(self) -> bool:
        # Loop over all parallel environments
        for i in range(len(self.locals['rewards'])):
            self.episode_total_rewards[i] += self.locals['rewards'][i]
            self.episode_pose_rewards[i] += self.locals['infos'][i]['pose_reward']
            self.episode_dis_rewards[i] += self.locals['infos'][i]['distance_reward']
            self.episode_success[i] += self.locals['infos'][i]['success_reward']
            self.episode_lengths[i] += 1

            # Check if the current episode has ended
            if self.locals['dones'][i]:
                self.episode_counts[i] += 1

                # Log average reward every `log_interval` episodes
                if self.episode_counts[i] % self.log_interval == 0:
                    avg_reward = self.episode_total_rewards[i] / self.log_interval
                    avg_pose_reward = self.episode_pose_rewards[i] / self.log_interval
                    avg_dis_reward = self.episode_dis_rewards[i] / self.log_interval
                    avg_success = self.episode_success[i] / self.log_interval

                    self.model.logger.record(f"reward/env_{i}", avg_reward, exclude="stdout")
                    self.model.logger.record(f"pose_reward/env_{i}", avg_pose_reward, exclude="stdout")
                    self.model.logger.record(f"distance_reward/env_{i}", avg_dis_reward, exclude="stdout")
                    self.model.logger.record(f"success_rate/env_{i}", avg_success, exclude="stdout")

                    self.model.logger.dump(step=self.episode_counts[i] / (self.log_interval - 1))

                    # Reset accumulators
                    self.episode_total_rewards[i] = 0.0
                    self.episode_pose_rewards[i] = 0.0
                    self.episode_dis_rewards[i] = 0.0
                    self.episode_success[i] = 0.0
                    self.episode_lengths[i] = 0

        return True

