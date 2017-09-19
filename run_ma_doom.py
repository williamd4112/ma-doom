#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from baselines.common.atari_wrappers import wrap_deepmind
from env import wrap_ma_doom
# from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from baselines.a2c.policies import MACnnPolicy

def train(env_id, num_frames, seed, policy, lrschedule, num_cpu):
    num_timesteps = int(num_frames / 4 * 1.1) 
    # divide by 4 due to frameskip, then do a little extras so episodes end
    def make_env(rank):
        def _thunk():
            gym.logger.setLevel(logging.WARN)
            return wrap_ma_doom()
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    if policy == 'cnn':
        policy_fn = MACnnPolicy
    elif policy == 'lstm':
        raise NotImplemented
    elif policy == 'lnlstm':
        raise NotImplemented
    learn(policy_fn, env, seed, total_timesteps=num_timesteps, lrschedule=lrschedule)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=40)
    args = parser.parse_args()
    train(args.env, num_frames=1e6 * args.million_frames, seed=args.seed, 
        policy=args.policy, lrschedule=args.lrschedule, num_cpu=16)

if __name__ == '__main__':
    main()
