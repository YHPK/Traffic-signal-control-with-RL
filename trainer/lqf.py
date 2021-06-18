import numpy as np

class Trainer:
    def __init__(self, args):
        self.max_step = args.max_step
        self.traffic_change_time = args.traffic_change_time

        self.rewards = []
        self.avg_relative_traffic_flow = []
        self.avg_cross_blocking = []

    def rollout(self, env):

        rtf = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        crb = 0
        episode_reward = 0
        obs = env.reset()
        action = 0

        for s in range(self.max_step):

            if s % self.traffic_change_time == 0:
                action = env.get_lqf_action()
                next_obs, rew, done = env.step(action)

            elif s % self.traffic_change_time == self.traffic_change_time - 1:
                next_obs, n_rew, done = env.step(action)
                rew += n_rew

                rtf += env.get_relative_traffic_flow()
                crb += env.get_avg_cross_block()
                episode_reward += rew
                obs = next_obs

            else:
                next_obs, n_rew, done = env.step(action)
                rew += n_rew

            # action = env.get_lqf_action()
            # next_obs, rew, done = env.step(action)
            # episode_reward += rew

        self.rewards.append(episode_reward)
        self.avg_relative_traffic_flow.append(rtf/self.max_step)
        self.avg_cross_blocking.append(crb/self.max_step)

        return episode_reward