import time
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        algo,
        eval_env,
        epoch,
        step_per_epoch,
        rollout_freq,
        logger,
        log_freq,
        eval_episodes=10
        # eval_episodes = 1
    ):
        self.algo = algo
        self.eval_env = eval_env

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._rollout_freq = rollout_freq

        self.logger = logger
        self._log_freq = log_freq
        self._eval_episodes = eval_episodes

    def train_dynamics(self):
        start_time = time.time()
        self.algo.learn_dynamics()
        #self.algo.save_dynamics_model(
            #save_path=os.path.join(self.logger.writer.get_logdir(), "dynamics_model")
        #)
        self.algo.save_dynamics_model("dynamics_model")
        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))

    def train_policy(self):
        start_time = time.time()
        num_timesteps = 0
        CVaR95 = []
        CVaR70 = []
        # train loop
        for e in range(1, self._epoch + 1):
            self.algo.policy.train()
            with tqdm(total=self._step_per_epoch, desc=f"Epoch #{e}/{self._epoch}") as t:
                while t.n < t.total:
                    if num_timesteps % self._rollout_freq == 0:
                        self.algo.rollout_transitions()
                    # update policy by sac
                    loss = self.algo.learn_policy()
                    t.set_postfix(**loss)
                    # log
                    if num_timesteps % self._log_freq == 0:
                        for k, v in loss.items():
                            self.logger.record(k, v, num_timesteps, printed=False)
                    num_timesteps += 1
                    t.update(1)
            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            ep_CVaR95_mean = np.mean(eval_info["eval/CVaR_95"])
            ep_CVaR70_mean = np.mean(eval_info["eval/CVaR_70"])
            
            CVaR95.append(ep_CVaR95_mean)
            CVaR70.append(ep_CVaR70_mean)
            self.logger.record("eval/episode_reward", ep_reward_mean, num_timesteps, printed=False)
            self.logger.record("eval/episode_length", ep_length_mean, num_timesteps, printed=False)
            self.logger.print(f"Epoch #{e}: episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f}, episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")
        
            # save policy
            torch.save(self.algo.policy.state_dict(), os.path.join(self.logger.writer.get_logdir(), "policy.pth"))
        self.logger.print("total time: {:.3f}s".format(time.time() - start_time))

        plt.plot(CVaR95, label='0.95-CVaR')
        plt.plot(CVaR70, label='0.7-CVaR')
        plt.xlabel('Epoch')
        plt.ylabel('CVaR')
        plt.legend()
        plt.show()

    def _evaluate(self):
        self.algo.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        profit_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.algo.policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action)
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            profit_buffer.append(reward)

            if terminal:
                CVaR_95 = self.calculate_cvar(profit_buffer, 0.05)
                CVaR_70 = self.calculate_cvar(profit_buffer, 0.30)
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length, "CVaR_95": CVaR_95, "CVaR_70":CVaR_70}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                profit_buffer = []
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer], 
            "eval/CVaR_95": [ep_info["CVaR_95"] for ep_info in eval_ep_info_buffer],
            "eval/CVaR_70": [ep_info["CVaR_70"] for ep_info in eval_ep_info_buffer]
        }
    
    def calculate_cvar(self, array, alpha):
        array = np.array(array)
        VaR = np.percentile(array, alpha * 100)
        CVaR = array[array <= VaR].mean()
        return CVaR
