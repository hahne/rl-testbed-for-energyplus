# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from gym import Env
from gym import spaces
from gym.utils import seeding
import sys, os, subprocess, time, signal, stat
from glob import glob
import gzip
import shutil
import random
import numpy as np
from scipy.special import expit
import pandas as pd
from argparse import ArgumentParser
from gym_energyplus.envs.pipe_io import PipeIo
from gym_energyplus.envs.energyplus_model import EnergyPlusModel
from gym_energyplus.envs.energyplus_build_model import build_ep_model

class SaveExpEnv(Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 energyplus_file=None,
                 model_file=None,
                 weather_file=None,
                 log_dir=None,
                 verbose=False):
        self.energyplus_process = None
        self.pipe_io = None
        
        # Verify path arguments
        if energyplus_file is None:
            energyplus_file = os.getenv('ENERGYPLUS')
        if energyplus_file is None:
            print('energyplus_env: FATAL: EnergyPlus executable is not specified. Use environment variable ENERGYPLUS.')
            return None
        if model_file is None:
            model_file = os.getenv('ENERGYPLUS_MODEL')
        if model_file is None:
            print('energyplus_env: FATAL: EnergyPlus model file is not specified. Use environment variable ENERGYPLUS_MODEL.')
            return None
        if weather_file is None:
            weather_file = os.getenv('ENERGYPLUS_WEATHER')
        if weather_file is None:
            print('energyplus_env: FATAL: EnergyPlus weather file is not specified. Use environment variable ENERGYPLUS_WEATHER.')
            return None
        if log_dir is None:
            log_dir = os.getenv('ENERGYPLUS_LOG')
        if log_dir is None:
            log_dir = 'log'
        
        # Initialize paths
        self.energyplus_file = energyplus_file
        self.model_file = model_file
        self.weather_files = weather_file.split(',')
        self.log_dir = log_dir
        
        # Create an EnergyPlus model
        self.ep_model = build_ep_model(model_file = self.model_file, log_dir = self.log_dir)

        self.action_space = self.ep_model.action_space
        self.observation_space = self.ep_model.observation_space
        # TODO: self.reward_space which defaults to [-inf,+inf]
        self.pipe_io = PipeIo()

        self.episode_idx = -1
        self.verbose = verbose

        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __del__(self):
        # In case you forget to call env.stop()
        pass
        
    def reset(self):

        self.episode_idx += 1
        self.timestep1 = 0
        self.ep_model.reset()
        return self.step(None)[0]

    
    def step(self, action):
        self.timestep1 += 1
        # Send action to the environment
        if action is not None:
            # baselines 0.1.6 changed action type
            if isinstance(action, np.ndarray) and isinstance(action[0], np.ndarray):
                action = action[0]
            self.ep_model.set_action(action)

        
        # Receive observation from the environment    
        # Note that in our co-simulation environment, the state value of the last time step can not be retrived from EnergyPlus process
        # because EMS framework of EnergyPlus does not allow setting EMS calling point ater the last timestep is completed.
        # To remedy this, we assume set_raw_state() method of each model handle the case raw_state is None.
        raw_state, done = self.receive_observation() # raw_state will be None for for call at total_timestep + 1
        self.ep_model.set_raw_state(raw_state)
        reward = self.ep_model.compute_reward()

        ot = random.uniform(-20, 50)
        wt = random.uniform(-20, 50)
        et = random.uniform(-20, 50)
        p1 = random.uniform(0, 1000000000.0)
        p2 = random.uniform(0, 1000000000.0)
        p3 = random.uniform(0, 1000000000.0)
        observation = np.array([ot, wt, et, p1, p2, p3])

        if self.timestep1 > 16000:
            done = True

        if done:
            print('EnergyPlusEnv: (done)')
        return observation, reward, done, {}
    

    def receive_observation(self):

        worst_west = 23.5
        worst_east = 23.5
        diff_west = 0.0
        diff_east = 0.0

        if self.ep_model.action is not None:
            slw = self.ep_model.action[0] + 3.5
            shw = self.ep_model.action[1]

            dlw = 0 if slw > 23.0 and slw < 23.5 else abs(23.25-slw)
            dhw = 0 if shw > 23.5 and shw < 24 else abs(23.75-shw)

            if dlw >= dhw and dlw >= diff_west:
                worst_west = slw
                diff_west = dlw
            if dhw >= dlw and dhw >= diff_west:
                worst_west = shw
                diff_west = dhw

            sle = self.ep_model.action[2]
            she = self.ep_model.action[3]

            dle = 0 if sle > 23.0 and sle < 23.5 else abs(23.25-sle)
            dhe = 0 if she > 23.5 and she < 24 else abs(23.75-she)

            if dle >= dhe and dle >= diff_east:
                worst_east = sle
                diff_east = dle
            if dhe >= dle and dhe >= diff_east:
                worst_east = she
                diff_east = dhe

        raw_state = [0.0, worst_west, worst_east, 0.0, 0.0, 0.0, 0.0]
        #raw_state = [0.0, 23.5, 23.5, 0.0, 0.0, 0.0, 0.0]
        return raw_state, False
    
    def render(self, mode='human'):
        if mode == 'human':
            return False
        
    def close(self):
        pass

    def plot(self, log_dir='', csv_file=''):
        self.ep_model.plot(log_dir=log_dir, csv_file=csv_file)

    def dump_timesteps(self, log_dir='', csv_file='', reward_file=''):
        self.ep_model.dump_timesteps(log_dir=log_dir, csv_file=csv_file)

    def dump_episodes(self, log_dir='', csv_file='', reward_file=''):
        self.ep_model.dump_episodes(log_dir=log_dir, csv_file=csv_file)
        
def parser():
    usage = 'Usage: python {} [--verbose] [--energyplus <file>] [--model <file>] [--weather <file>] [--simulate] [--plot] [--help]'.format(__file__)
    argparser = ArgumentParser(usage=usage)
    #argparser.add_argument('fname', type=str,
    #                       help='echo fname')
    argparser.add_argument('-v', '--verbose',
                           action='store_true',
                           help='Show verbose message')
    argparser.add_argument('-e', '--energyplus', type=str,
                           dest='energyplus',
                           help='EnergyPlus executable file')
    argparser.add_argument('-m', '--model', type=str,
                           dest='model',
                           help='Model file')
    argparser.add_argument('-w', '--weather', type=str,
                           dest='weather',
                           help='Weather file')
    argparser.add_argument('-s', '--simulate',
                           action='store_true',
                           help='Do simulation')
    argparser.add_argument('-p', '--plot',
                           action='store_true',
                           help='Do plotting')
    return argparser.parse_args()

def easy_agent(next_state, target, hi, lo):
    sensitivity_pos = 1.0
    sensitivity_neg = 1.0
    act_west_prev = 0
    act_east_prev = 0
    alpha = 0.4

    delta_west = next_state[1] - target
    if delta_west >= 0:
        act_west = target - delta_west * sensitivity_pos
    else:
        act_west = target - delta_west * sensitivity_neg
    act_west = act_west * alpha + act_west_prev * (1 - alpha)
    act_west_prev = act_west
    
    delta_east = next_state[2] - target
    if delta_east >= 0:
        act_east = target - delta_east * sensitivity_pos
    else:
        act_east = target - delta_east * sensitivity_neg
    act_east = act_east * alpha + act_east_prev * (1 - alpha)
    act_east_prev = act_east

    act_west = max(lo, min(act_west, hi))
    act_east = max(lo, min(act_east, hi))
    action = np.array([act_west, act_west, act_west, act_west, act_east, act_east, act_east, act_east])
    return action



if __name__ == '__main__':

    args = parser()
    print('args={}'.format(args))
    
    lo = 0.0
    hi = 40.0
    target = 23.0
    
    # obs[0]: Eronment:Site Outdoor Air Drybulb Temperature [C](TimeStep)
    # obs[1]: Workload level (not implemented yet)
    #obs_space = spaces.Box(np.array([-20.0, 0.0]),
    #                       np.array([ 50.0, 1.0]))

    # act[0]: WestZoneDECOutletNode_setpoint
    # act[1]: WestZoneIECOutletNode_setpoint
    # act[2]: WestZoneCCoilAirOutletNode_setpoint
    # act[3]: WestAirLoopOutletNode_setpoint
    # act[4]: EastZoneDECOutletNode_setpoint
    # act[5]: EastZoneIECOutletNode_setpoint
    # act[6]: EastZoneCCoilAirOutletNode_setpoint
    # act[7]: EastAirLoopOutletNode_setpoint
    #act_space = spaces.Box(np.array([ lo, lo, lo, lo, lo, lo, lo, lo]),
    #                       np.array([ hi, hi, hi, hi, hi, hi, hi, hi]))
        
    # just for testing
    env = SaveExpEnv(verbose = args.verbose)
    if env is None:
        quit()

    if (args.simulate):
        for ep in range(1):
            PUE_min = 100.
            PUE_max = 0.
            PUE_sum = 0.
            PUE_count = 0
            next_state = env.reset()

            for i in range(1000000):
                #if args.verbose:
                #    os.system('clear')
                #    print('Step {}'.format(i))
                    
                #action = env.action_space.sample()
                action = easy_agent(next_state, target, hi, lo)
                PUE = next_state[3]
                PUE_sum += PUE
                PUE_min = min(PUE, PUE_min)
                PUE_max = max(PUE, PUE_max)
                PUE_count += 1

                next_state, reward, done, _ = env.step(action)
                PUE_ave = PUE_sum / PUE_count

                if args.verbose:
                    print('========= count={} PUE={} PUEave={} PUEmin={} PUEmax={}'.format(PUE_count, PUE, PUE_ave, PUE_min, PUE_max))
                if done:
                    break
            PUE_ave = PUE_sum / PUE_count
            print('============================= Episodo done. count={} PUEave={} PUEmin={} PUEmax={}'.format(PUE_count, PUE_ave, PUE_min, PUE_max))
            #env.close()

    env.plot()
