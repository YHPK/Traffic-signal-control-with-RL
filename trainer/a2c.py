from models import a2c
import torch
import torch.optim as optim
import numpy as np

class Trainer:
    def __init__(self, obs_dim, n_actions, args):
        self.max_step = args.max_step
        self.traffic_change_time = args.traffic_change_time
        self.device = args.device
        self.gamma = args.gamma

        self.policy_model = a2c.Policy(obs_dim=obs_dim, n_actions=n_actions, device=args.device)
        self.policy_model.to(args.device)
        self.optimizer = optim.Adam(self.policy_model.parameters())

        self.rewards = []
        self.avg_relative_traffic_flow = []
        self.avg_cross_blocking = []

    def rollout(self, env):

        logprobs = []
        returns = []
        values = []

        rtf = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        crb = 0
        episode_reward = 0
        obs = env.reset()
        action = 0

        for s in range(self.max_step):

            if s % self.traffic_change_time == 0:
                action, logprob, value = self.policy_model.get_action_logprob_value(obs)
                next_obs, rew, done = env.step(action)


            elif s % self.traffic_change_time == self.traffic_change_time -1:
                next_obs, n_rew, done = env.step(action)
                rew += n_rew

                logprobs.append(logprob)
                returns.append(rew)
                values.append(value)

                rtf += env.get_relative_traffic_flow()
                crb += env.get_avg_cross_block()
                episode_reward += rew
                obs = next_obs

                # os.system('clear')
                # print("episode:", e, "step count:", s, "reward:", episode_reward, end='\r')
                # env.show_status()

            else:
                next_obs, n_rew, done = env.step(action)
                rew += n_rew


        # Train
        for i in range(len(returns) - 2, -1, -1):
            returns[i] += returns[i + 1] * self.gamma
        returns = torch.tensor(returns).to(self.device)
        a2c.train(logprobs, returns, values, self.optimizer)

        self.avg_relative_traffic_flow.append(rtf/self.max_step)
        self.avg_cross_blocking.append(crb/self.max_step)
        self.rewards.append(episode_reward)

        return episode_reward