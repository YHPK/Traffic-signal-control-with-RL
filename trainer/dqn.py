from models import dqn
from util import ReplayBuffer
import numpy as np
import torch
import torch.optim as optim


class Trainer:
    def __init__(self, obs_dim, n_actions, args):
        self.max_step = args.max_step
        self.traffic_change_time = args.traffic_change_time
        self.device = args.device
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.batch_size = args.batch_size

        self.copy_freq = 200
        buffer_size = 2000
        obs_shape = tuple([40])
        act_shape = tuple([1])


        self.q_model = dqn.Policy(obs_dim=obs_dim, n_actions=n_actions, device=self.device)
        self.target_q_model = dqn.Policy(obs_dim=obs_dim, n_actions=n_actions, device=self.device)
        self.q_model.to(self.device)
        self.target_q_model.to(self.device)

        self.target_q_model.load_state_dict(self.q_model.state_dict())

        self.replay_buffer = ReplayBuffer(obs_shape, act_shape, buffer_size)
        self.optimizer = optim.Adam(self.q_model.parameters())

        self.rewards = []
        self.td_errors = []
        self.avg_relative_traffic_flow = []
        self.avg_cross_blocking = []


    def rollout(self, env):

        rtf = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        crb = 0
        episode_reward = 0
        obs = env.reset()

        for s in range(self.max_step):

            if s % self.traffic_change_time == 0:
                action = self.q_model.get_action(obs, self.epsilon)
                next_obs, rew, done = env.step(action)

            elif s % self.traffic_change_time == self.traffic_change_time - 1:
                next_obs, n_rew, done = env.step(action)
                rew += n_rew
                self.replay_buffer.store(obs, action, rew, next_obs, done)

                rtf += env.get_relative_traffic_flow()
                crb += env.get_avg_cross_block()
                episode_reward += rew

                obs = next_obs

            else:
                next_obs, n_rew, done = env.step(action)
                rew += n_rew

            if done:
                # print("episode_reward :", episode_reward)
                obs = env.reset()
                episode_reward = 0

            # Train
            if len(self.replay_buffer) > self.batch_size:
                sampled_batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
                td_error = dqn.train(self.q_model, self.target_q_model, sampled_batch, self.optimizer, self.gamma, self.device)
                if s % 50 == 0:
                    self.td_errors.append(td_error)

            if s % self.copy_freq == 0:
                self.target_q_model.load_state_dict(self.q_model.state_dict())


        self.avg_relative_traffic_flow.append(rtf/self.max_step)
        self.avg_cross_blocking.append(crb/self.max_step)
        self.rewards.append(episode_reward)

        return episode_reward





