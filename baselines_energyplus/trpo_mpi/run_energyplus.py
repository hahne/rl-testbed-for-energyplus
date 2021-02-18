#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines_energyplus.common.energyplus_util import make_energyplus_env, energyplus_arg_parser, energyplus_logbase_dir
from stable_baselines import logger
#from stable_baselines.common.models import mlp
from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.ppo1.mlp_policy import MlpPolicy
from stable_baselines.trpo_mpi import trpo_mpi
from stable_baselines import TRPO
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import tensorflow as tf, numpy as np
import os
import shutil
import datetime
import gym_energyplus

def policy_fn(name, ob_space, ac_space):
    return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=32, num_hid_layers=2)

def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    #def policy_fn(name, ob_space, ac_space):
    #    return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    #        hid_size=32, num_hid_layers=2)

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
        logger.configure(log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    env = make_energyplus_env(env_id, workerseed)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env)

    model = TRPO(MlpPolicy, env, verbose=1,
            #timesteps_per_batch=1*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
            timesteps_per_batch=16*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
            gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    model.learn(total_timesteps=num_timesteps)

    # trpo_mpi.learn(env, policy_fn,
    #                max_timesteps=num_timesteps,
    #                #timesteps_per_batch=1*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
    #                timesteps_per_batch=16*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
    #                gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)

    # trpo_mpi.learn(env=env, network=mlp(num_hidden=32, num_layers=2),
    #                total_timesteps=num_timesteps,
    #                #timesteps_per_batch=1*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
    #                timesteps_per_batch=16*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
    #                gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)

    # run_mpi.learn(env, policy_fn,
    #                max_timesteps=num_timesteps, timesteps_per_batch=16 * 1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
    #                 gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)

    env.close()

    model.save(log_dir + '/model')

def main():
    args = energyplus_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()

