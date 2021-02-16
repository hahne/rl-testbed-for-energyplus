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

    ac = env.action_space.sample()
    ob = env.reset()
    ac = np.array([-0.8, -0.8, 1.0, 1.0])
    for iter in range(num_timesteps):

        if ob[1] > 23.6:
            ac[0] -= 0.01
            ac[2] += 0.05
        if ob[1] < 23.4:
            ac[0] += 0.01
            ac[2] -= 0.05
        
        if ob[2] > 23.6:
            ac[1] -= 0.01
            ac[3] += 0.05
        if ob[2] < 23.4:
            ac[1] += 0.01
            ac[3] -= 0.05

        ob, rew, done, _ = env.step(ac)

        #print(ob)

        if done:
            ob = env.reset()


    env.close()

def main():
    args = energyplus_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()

