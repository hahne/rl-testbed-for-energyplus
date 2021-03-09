#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "text.usetex": True,
    "pgf.preamble": "\n".join([
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
    ]),
})

from baselines_energyplus.common.energyplus_util import make_energyplus_env, energyplus_locate_log_dir
import gym_energyplus

def plot_energyplus_arg_parser():
    """
    Create an argparse.ArgumentParser for plot_energyplus.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', '-e', help='environment ID', type=str, default='EnergyPlus-v0')
    parser.add_argument('--log-dirs', '-l', help='Plot all data in the log directory', type=str, default='')
    #parser.add_argument('--data', '-m', help='Plot data', type=str, default='reward')
    return parser

def energyplus_plot(env_id, log_dirs=''):
    env = make_energyplus_env(env_id, 0)

    log_dirs_list = log_dirs.split(',')
    print(log_dirs_list)

    rewards = []
    powers = []
    temp_w = []
    temp_e = []
    labels = []
    num_episodes = 200
    for log_dir in log_dirs_list:
        label = re.search('\d\d\d\d-\d\d-\d\d-\d\d-\d\d-\d\d', log_dir).group(0)
        labels.append(label)
        env.ep_model.get_episode_list(log_dir)
        print(env.ep_model.num_episodes)
        rewards_per_episode = []
        total_power_per_episode = []
        temp_w_per_episode = []
        temp_e_per_episode = []
        for ep in range(min([env.ep_model.num_episodes, num_episodes])):
            print(ep)
            print('Episode {}'.format(ep))
            env.ep_model.read_episode(ep)
            MeanRew, _, _, _ = env.ep_model.get_statistics(env.ep_model.rewards)
            print('MeanRew: ' + str(MeanRew))
            rewards_per_episode.append(MeanRew)

            MeanPower, _, _, _ = env.ep_model.get_statistics(env.ep_model.total_electric_demand_power / 1000.0)
            print('MeanPower: ' + str(MeanPower))
            total_power_per_episode.append(MeanPower)

            west_in_range = ((22 < env.ep_model.westzone_temp) & (env.ep_model.westzone_temp < 25)).sum() / len(env.ep_model.westzone_temp)
            print('west_in_range: ' + str(west_in_range))
            temp_w_per_episode.append(west_in_range)

            east_in_range = ((22 < env.ep_model.eastzone_temp) & (env.ep_model.eastzone_temp < 25)).sum() / len(env.ep_model.eastzone_temp)
            print('east_in_range: ' + str(east_in_range))
            temp_e_per_episode.append(east_in_range)

        rewards.append(rewards_per_episode)
        powers.append(total_power_per_episode)
        temp_w.append(temp_w_per_episode)
        temp_e.append(temp_e_per_episode)


    #labels = ['Rule-based + RLv2', 'RLv2', 'RLv1']
    labels[0] = "Baseline"
    labels[1] = "Moriyama et al."
    labels[2] = "Rule-based"
    labels[3] = "Rule-based + RL"
    labels[4] = "Rule-based + RL (temp rew max 0.8)"

    idx = 0
    for rewards_per_episode in rewards:
        rewards_per_episode = np.pad(rewards_per_episode,(0,max(0,num_episodes-len(rewards_per_episode))),mode='edge')
        plt.plot(range(num_episodes),rewards_per_episode, label=labels[idx])
        plt.title('Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        idx +=1

    plt.legend()
    #plt.show()
    plt.savefig('reward.pdf', bbox_inches='tight', pad_inches = 0.02)
    plt.savefig('reward.pgf', bbox_inches='tight', pad_inches = 0.02)
    plt.close()


    idx = 0
    for powers_per_episode in powers:
        powers_per_episode = np.pad(powers_per_episode,(0,max(0,num_episodes-len(powers_per_episode))),mode='edge')
        plt.plot(range(num_episodes),powers_per_episode, label=labels[idx])
        plt.title('Power Consumption over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Power in kW')
        idx +=1

    plt.legend()
    #plt.show()
    plt.savefig('power_consumption.pdf', bbox_inches='tight', pad_inches = 0.02)
    plt.savefig('power_consumption.pgf', bbox_inches='tight', pad_inches = 0.02)
    plt.close()

    idx = 0
    for temp_w_per_episode in temp_w:
        temp_w_per_episode = np.pad(temp_w_per_episode,(0,max(0,num_episodes-len(temp_w_per_episode))),mode='edge')
        plt.plot(range(num_episodes),temp_w_per_episode, label=labels[idx])
        plt.title('West Zone Temperature in Range')
        plt.xlabel('Episode')
        plt.ylabel('timesteps in range / total')
        idx +=1

    plt.legend()
    #plt.show()
    plt.savefig('west_zone_in_range.pdf', bbox_inches='tight', pad_inches = 0.02)
    plt.savefig('west_zone_in_range.pgf', bbox_inches='tight', pad_inches = 0.02)
    plt.close()

    idx = 0
    for temp_e_per_episode in temp_e:
        temp_e_per_episode = np.pad(temp_e_per_episode,(0,max(0,num_episodes-len(temp_e_per_episode))),mode='edge')
        plt.plot(range(num_episodes),temp_e_per_episode, label=labels[idx])
        plt.title('East Zone Temperature in Range')
        plt.xlabel('Episode')
        plt.ylabel('timesteps in range / total')
        idx +=1

    plt.legend()
    #plt.show()
    plt.savefig('east_zone_in_range.pdf', bbox_inches='tight', pad_inches = 0.02)
    plt.savefig('east_zone_in_range.pgf', bbox_inches='tight', pad_inches = 0.02)
    plt.close()

    env.close()

def main():
    args = plot_energyplus_arg_parser().parse_args()
    energyplus_plot(args.env, log_dirs=args.log_dirs)

if __name__ == '__main__':
    main()
