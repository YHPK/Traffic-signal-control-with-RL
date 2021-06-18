from env import FiveIntersection
from trainer import a2c_Trainer, lqf_Trainer, dqn_Trainer
from util import plot, logger

import argparse
import pickle
import torch

def main(args):

    env = FiveIntersection(args)
    obs_dim = env.observation_space_size
    n_actions = env.action_space_size

    if args.algo_type == 'lqf':
        trainer = lqf_Trainer(args)

    elif args.algo_type == 'dqn':
        trainer = dqn_Trainer(obs_dim, n_actions, args)

    elif args.algo_type == 'a2c':
        trainer = a2c_Trainer(obs_dim, n_actions, args)

    else:
        print('wrong algo type')
        exit(-1)

    rew_path = f'{args.log_save_path}/{args.algo_type}_reward'
    rtf_path = f'{args.log_save_path}/{args.algo_type}_rtf'
    crb_path = f'{args.log_save_path}/{args.algo_type}_crb'

    for e in range(args.episode):

        episode_reward = trainer.rollout(env)
        print("episode:", e, "reward:", episode_reward)
        # print("avg cross block:", trainer.avg_cross_blocking)
        # print("avg relative traffic flow:", trainer.avg_relative_traffic_flow)

        if e % 10 == 0:


            plot(trainer.rewards, rew_path)
            plot(trainer.avg_relative_traffic_flow, rtf_path)
            plot(trainer.avg_cross_blocking, crb_path)

            pickle.dump(trainer.rewards, open(rew_path+'.pkl', 'wb'))
            pickle.dump(trainer.avg_relative_traffic_flow, open(rtf_path+'.pkl', 'wb'))
            pickle.dump(trainer.avg_cross_blocking, open(crb_path+'.pkl', 'wb'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--episode', type=int, default=1500)
    parser.add_argument('--max_step', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0.05)

    parser.add_argument('--traffic_change_time', type=int, default=20)
    parser.add_argument('--average_arrival_time', type=float, default=0.1)
    parser.add_argument('--algo_type', type=str, default='lqf', choices=['lqf', 'dqn', 'a2c'])
    parser.add_argument('--render_type', type=str, default=None, choices=['print', 'draw'])
    parser.add_argument('--log_save_path', type=str, default='.')

    args = parser.parse_args()
    args.device = 'cuda:{}'.format(args.gpu_num) if torch.cuda.is_available() else 'cpu'

    print('device type:', args.device)

    main(args)
