#!/usr/bin/env python
import os, logging, gym, time
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from a2c import learn
#from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from subproc_env import SubprocVecEnv
# from baselines.common.atari_wrappers import wrap_deepmind
from env import wrap_ma_doom
# from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from policies import MACommPolicy, MACnnPolicy, MAReconPolicy

NUM_PLAYERS = 2

def train(config, num_frames, seed, policy, lrschedule, num_cpu, ckpt, nsteps, start_port=8000, dfn=all):
    num_timesteps = int(num_frames / 4 * 1.1)
    # divide by 4 due to frameskip, then do a little extras so episodes end
    def make_env(rank):
        def _thunk():
            port = rank + start_port
            gym.logger.setLevel(logging.WARN)
            return wrap_ma_doom(config, NUM_PLAYERS, port)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], dfn)
    if policy == 'comm':
        policy_fn = MACommPolicy
    elif policy == 'commsep':
        policy_fn = MACommSepCriticPolicy
    elif policy == 'cnn':
        policy_fn = MACnnPolicy
    elif policy == 'recon':
        policy_fn = MAReconPolicy
    elif policy == 'lnlstm':
        raise NotImplemented
    time.sleep(num_cpu * 1)
    print("creation complete, start running!")
    return learn(policy_fn, env, seed, nsteps=nsteps, checkpoint=ckpt, total_timesteps=num_timesteps, lrschedule=lrschedule)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--any', help='reset after any players die', action='store_true')
    parser.add_argument('--config', help='config path', default='data/triple_lines_easy.cfg')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--ckpt', help='idx of ckpt', type=int, default=0)
    parser.add_argument('--nsteps', help='num steps per update', type=int, default=8)
    parser.add_argument('--ncpu', help='num cpu', type=int, default=8)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'comm', 'commsep', 'recon'], default='comm')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--thousand_frames', help='How many frames to train (/ 1e3). '
        'This number gets divided by 4 due to frameskip', type=int, default=600)
    parser.add_argument('--no-recon', action='store_true', help='merge dec into fc ')
    parser.add_argument('--port', type=int, default=8000, help='merge dec into fc ')
    args = parser.parse_args()
    print(args)
    current_ckpt = args.ckpt
    dfn = any if args.any else all
    while True:
        current_ckpt = train(args.config, num_frames=1e3 * args.thousand_frames, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_cpu=args.ncpu, ckpt=current_ckpt, nsteps=args.nsteps, dfn=dfn, start_port=args.port)
        time.sleep(3)

if __name__ == '__main__':
    main()
