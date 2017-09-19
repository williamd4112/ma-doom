from vizdoom_map.ma_doom_env import DoomSyncMultiPlayerEnvironment

import gym
from gym import error, spaces


class GymDoomSyncMultiPlayerEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config, num_players):
        self.num_players = num_players
        self.doom_env = DoomSyncMultiPlayerEnvironment(config, num_players) 
        
        # Get action space from first environment
        # For now, we assume each agent has the same action space
        self.action_space = spaces.Discrete(len(self.doom_env.actions[0]))
       
        # Get observation space from first environment
        # For now, we assume observation shape is fixed as (320, 240)
        self.observation_space = spaces.Box(low=0, high=255, shape=(320, 240, 3))
    
    def _reset(self):
        self.doom_env.reset()
        return self.doom_env.current_state()

    def _step(self, a):
        reward = 0.0
        info = {}
        rewards, done = self.doom_env.step(a)
        next_state = self.doom_env.current_state()

        return next_state, rewards, done, info

if __name__ == '__main__':
    env = GymDoomSyncMultiPlayerEnvironment('data/coop.cfg', 2)
    env.reset()
    for episode in range(100):
        print('Episode %d' % (episode))
        act = [ env.action_space.sample(), env.action_space.sample()]
        next_state, rewards, done, _ = env.step(act)
        print(next_state[0].shape, rewards, done)
        if done:
            env.reset()
