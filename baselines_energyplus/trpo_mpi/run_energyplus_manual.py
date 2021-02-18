#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines_energyplus.common.energyplus_util import make_energyplus_env, energyplus_arg_parser, energyplus_logbase_dir
from baselines import logger
import os
import shutil
import datetime
import gym_energyplus
import numpy as np
import gym

def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

    # Create a new base directory like /tmp/openai-2018-05-21-12-27-22-552435
    log_dir = os.path.join(energyplus_logbase_dir(), datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    if not os.path.exists(log_dir + '/output'):
        os.makedirs(log_dir + '/output')
    os.environ["ENERGYPLUS_LOG"] = log_dir
    model = os.getenv('ENERGYPLUS_MODEL')
    if model is None:
        print('Environment variable ENERGYPLUS_MODEL is not defined')
        os.exit()
    weather = os.getenv('ENERGYPLUS_WEATHER')
    if weather is None:
        print('Environment variable ENERGYPLUS_WEATHER is not defined')
        os.exit()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print('train: init logger with dir={}'.format(log_dir)) #XXX
        logger.configure(log_dir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    env = make_energyplus_env(env_id, workerseed)

    n_episodes = 20
    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []
    ep_idx = 0
    episode_starts.append(True)
    reward_sum = 0.0

    ac = env.action_space.sample()
    ob = env.reset()
    ac = np.array([-0.7, -0.7, 1.0, 1.0])

    while ep_idx < n_episodes:
        #for iter in range(num_timesteps):

        if ob[1] > 23.6:
            ac[0] -= 0.01
            ac[2] -= 0.03
        if ob[1] < 23.4:
            ac[0] += 0.01
            ac[2] += 0.03
        if ob[2] > 23.6:
            ac[1] -= 0.01
            ac[3] -= 0.03
        if ob[2] < 23.4:
            ac[1] += 0.01
            ac[3] += 0.03

        # ac[2] = 1.0
        # ac[3] = 1.0
        # if ob[1] > 23.0 and ob[1] < 24.0:
        #     ac[2] = 0.9
        # if ob[2] > 23.0 and ob[2] < 24.0:
        #     ac[3] = 0.9
        # if ob[1] > 23.25 and ob[1] < 23.75:
        #     ac[2] = 0.8
        # if ob[2] > 23.25 and ob[2] < 23.75:
        #     ac[3] = 0.8

        #ac_clipped = np.clip(ac, a_min =[-1.0, -1.0, 0.3, 0.3], 
        #                        a_max = [-0.2, -0.2, 1.0, 1.0]) 
        ob, rew, done, _ = env.step(ac)

        #print(ob)
        observations.append(ob)
        actions.append(ac)
        rewards.append(rew)
        episode_starts.append(done)
        reward_sum += rew
        if done:
            ob = env.reset()

            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1

    if isinstance(env.observation_space, gym.spaces.Box):
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, gym.spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))

    if isinstance(env.action_space, gym.spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }  # type: Dict[str, np.ndarray]

    for key, val in numpy_dict.items():
        print(key, val.shape)

    save_path = os.path.join(log_dir, "base_controller.npz")
    if save_path is not None:
        np.savez(save_path, **numpy_dict)

    env.close()

def main():
    args = energyplus_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()

