from vizdoom_map.ma_doom_env import DoomSyncMultiPlayerEnvironment
from pygame_rl.scenario.predator_prey_environment import PredatorPreyEnvironment, PredatorPreyEnvironmentOptions
import numpy as np
import random
import time
import logging

from collections import deque
import gym
from gym import error, spaces

import cv2
import sys


class MockGymDoomSyncMultiPlayerEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config, num_players, port):
        self.num_players = num_players
        #self.doom_env = DoomSyncMultiPlayerEnvironment(config, num_players)

        # Get action space from first environment
        # For now, we assume each agent has the same action space
        self.action_space = spaces.Discrete(5)

        # Get observation space from first environment
        # For now, we assume observation shape is fixed as (320, 240)
        self.observation_space = spaces.Box(low=0, high=255, shape=(320, 240, 3))

    def _reset(self):
        # self.doom_env.reset()
        return [np.random.rand(*[320, 240, 3])] * self.num_players

    def _step(self, a):
        info = {}
        #rewards, done = self.doom_env.step(a)
        rewards = [np.random.rand()] * self.num_players
        done = random.choice([True, False])
        #next_states = self.doom_env.current_state()
        next_states = [np.random.rand(*[320, 240, 3])] * self.num_players

        return next_states, rewards, done, info


class GymDoomSyncMultiPlayerEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config, num_players, port=8000):
        self.num_players = num_players
        self.doom_env = DoomSyncMultiPlayerEnvironment(config, num_players, port, deathmatch=False)

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
        info = {}
        rewards, done = self.doom_env.step(a)
        next_states = self.doom_env.current_state()

        return next_states, rewards, done, info

    def _close(self):
        self.doom_env.close()


class GymPredatorPreySyncMultiPlayerEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    act_map = ["MOVE_RIGHT", "MOVE_UP", "MOVE_LEFT", "MOVE_DOWN", "STAND"]

    def __init__(self, env_options, res=84, step_penalty=0, po_radius=3):
        # Create the environment
        self.env = PredatorPreyEnvironment(env_options=env_options)
        self.res = res
        self.po_radius = po_radius
        self.step_penalty = step_penalty

        # Get action space
        self.action_space = spaces.Discrete(len(self.env.actions))

        # Render the environment
        self.env.render()

        # Get observation space
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))
        self.predator_idx_rng = self.env.get_group_index_range("PREDATOR")

    def _reset(self):
        self.env.reset()
        return self._proc_obs(self.env.state)

    def _step(self, a):
        for i, action in enumerate(a):
            self.env.take_cached_action(i, self.act_map[action])
        env_obs = self.env.update_state()
        obs = self.env.state
        reward = env_obs.reward if env_obs.reward != 0 else self.step_penalty
        done = obs.is_terminal()
        obs = self._proc_obs(obs)
        info = {}

        return obs, reward, done, info

    def _proc_obs(self, obs):
        screens = []
        positions = []
        for idx in range(*self.predator_idx_rng):
            pos = np.array(obs.get_object_pos(idx))
            screen = cv2.cvtColor(self.env.renderer.get_po_screenshot(pos, self.po_radius), cv2.COLOR_BGR2GRAY)
            screens.append(cv2.resize(screen, (self.res, self.res)))
            positions.append(pos)
        screens = [screen.reshape((self.res, self.res, 1)) for screen in screens]
        obs = list(zip(screens, positions))

        return obs


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, res=84, grayscale=True):
        self.grayscale = grayscale
        self.channel = 3 if not grayscale else 1
        gym.ObservationWrapper.__init__(self, env)
        self.res = res
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, self.channel))

    def _observation(self, obs):
        frames = []
        for i, frame in enumerate(obs):
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(frame, (self.res, self.res)))
        #frames = [cv2.resize(frame, (self.res, self.res)) for frame in obs]
        return [frame.reshape((self.res, self.res, self.channel)) for frame in frames]


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, nplayers, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip
        self.nplayers = nplayers

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = np.zeros(self.nplayers)
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class NdarrayEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def _state_to_ndarray(self, states):
        states_expand = [state[np.newaxis, ::] for state in states]
        return np.concatenate(states_expand)

    def _reward_to_ndarray(self, rewards):
        return np.asarray(rewards)

    def _step(self, action):
        next_states, rewards, done, info = self.env.step(action)
        return self._state_to_ndarray(next_states), self._reward_to_ndarray(rewards), done, info

    def _reset(self):
        states = self.env.reset()
        return self._state_to_ndarray(states)


def wrap_ma_doom(config, nplayers, port):
    env = GymDoomSyncMultiPlayerEnvironment(config, nplayers, port)
    env = WarpFrame(env)
    env = NdarrayEnv(env)
    env = MaxAndSkipEnv(env, nplayers)
    return env

def wrap_predator_prey(map_path, frame_skip=2, npredator=3, nprey=3, nobstacle=8, po_radius=2):
    object_size = {
        "PREDATOR": npredator,
        "PREY": nprey,
        "OBSTACLE": nobstacle
    }
    env_options = PredatorPreyEnvironmentOptions(
                        map_path=map_path,
                        object_size=object_size,
                        ai_frame_skip=frame_skip
                  )
    env =  GymPredatorPreySyncMultiPlayerEnvironment(env_options, po_radius=po_radius)

    return env

def main():
    import argparse
    config = argparse.ArgumentParser()
    config.parse_args()
    config.npredator = 3
    config.nprey = 3
    config.nobstacle = 8
    config.map_path = "data/map/predator_prey/predator_prey_15x15.tmx"
    config.frame_skip = 2
    config.res = 84
    env = wrap_predator_prey(config)
    env.reset()

    for i in range(10):
        obs, reward, done, info = env.step([0, 0, 0])
        for o in obs:
            print(o[0].shape)
            print(o[1])



if __name__ == '__main__':
    main()
