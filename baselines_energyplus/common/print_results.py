#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np

from baselines_energyplus.common.energyplus_util import make_energyplus_env, energyplus_locate_log_dir
import gym_energyplus

def plot_energyplus_arg_parser():
    """
    Create an argparse.ArgumentParser for plot_energyplus.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', '-e', help='environment ID', type=str, default='EnergyPlus-v0')
    parser.add_argument('--log-dir', '-l', help='Plot all data in the log directory', type=str, default='')
    parser.add_argument('--episode', '-i', help='Episode', type=str, default='0')
    return parser

def energyplus_print(env_id, log_dir='', ep_id=0):
    env = make_energyplus_env(env_id, 0)

    env.ep_model.get_episode_list(log_dir)
    print('Num Episodes {}'.format(env.ep_model.num_episodes))
    print('Episode {}'.format(ep_id))
    env.ep_model.read_episode(int(ep_id))
    MeanRew, _, _, _ = env.ep_model.get_statistics(env.ep_model.rewards)
    print('MeanRew: ' + str(MeanRew))

    MeanPower, _, _, _ = env.ep_model.get_statistics(env.ep_model.total_electric_demand_power / 1000.0)
    print('MeanPower: ' + str(MeanPower))

    west_in_range = ((22 < env.ep_model.westzone_temp) & (env.ep_model.westzone_temp < 25)).sum() / len(env.ep_model.westzone_temp)
    print('west_in_range: ' + str(west_in_range))

    east_in_range = ((22 < env.ep_model.eastzone_temp) & (env.ep_model.eastzone_temp < 25)).sum() / len(env.ep_model.eastzone_temp)
    print('east_in_range: ' + str(east_in_range))

   
def main():
    args = plot_energyplus_arg_parser().parse_args()
    energyplus_print(args.env, log_dir=args.log_dir, ep_id=args.episode)

if __name__ == '__main__':
    main()
