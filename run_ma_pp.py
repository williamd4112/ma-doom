#!/usr/bin/env python
import os, logging, gym, time
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from a2c import learn
#from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from subproc_env import SubprocVecEnv
# from baselines.common.atari_wrappers import wrap_deepmind
from env import wrap_predator_prey
# from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from policies import MANMapPolicy, MACnnPolicy

NUM_PLAYERS = 2

def train(config, num_frames, seed, policy, lrschedule, num_cpu, ckpt, nsteps, dfn=all):
    num_timesteps = int(num_frames / 4 * 1.1)
    # divide by 4 due to frameskip, then do a little extras so episodes end
    def make_env(rank):
        def _thunk():
            gym.logger.setLevel(logging.WARN)
            return wrap_predator_prey(**config)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], dfn)
    if policy == 'nmap':
        policy_fn = MANMapPolicy
    elif policy == 'cnn':
        policy_fn = MACnnPolicy
    elif policy == 'lnlstm':
        raise NotImplemented
    time.sleep(num_cpu * 1)
    print("creation complete, start running!")
    return learn(policy_fn, env, seed, nplayers=config["npredator"],
            nsteps=nsteps, checkpoint=ckpt, total_timesteps=num_timesteps, lrschedule=lrschedule)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--map-path', help='map path', default='data/map/predator_prey/predator_prey_15x15.tmx')
    parser.add_argument('--npredator', help='number of predators', type=int, default=3)
    parser.add_argument('--nprey', help='number of predators', type=int, default=3)
    parser.add_argument('--nobstacle', help='number of predators', type=int, default=8)
    parser.add_argument('--frame_skip', help='number of predators', type=int, default=1)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--ckpt', help='idx of ckpt', type=int, default=0)
    parser.add_argument('--nsteps', help='num steps per update', type=int, default=8)
    parser.add_argument('--ncpu', help='num cpu', type=int, default=8)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'nmap'], default='nmap')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--thousand_frames', help='How many frames to train (/ 1e3). '
        'This number gets divided by 4 due to frameskip', type=int, default=600)
    args = parser.parse_args()
    print(args)
    current_ckpt = args.ckpt
    config = {
                "map_path": args.map_path,
                "nprey": args.nprey,
                "npredator": args.npredator,
                "nobstacle": args.nobstacle,
                "frame_skip": args.frame_skip
    }

    while True:
        current_ckpt = train(config, num_frames=1e3 * args.thousand_frames, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_cpu=args.ncpu, ckpt=current_ckpt, nsteps=args.nsteps)
        time.sleep(3)

if __name__ == '__main__':
    main()
