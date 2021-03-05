from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager

import time

class PID:
    """PID Controller
    """

    def __init__(self, P=0.11, I=0.01, D=0.2, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback

        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}

        .. figure:: images/pid_1.png
           :align:   center

           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)

        """
        error = self.SetPoint - feedback_value

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = 1#self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time


def manual_controller(ac, ob, flow_baseline=0.0):
    if ob[1] > 23.6:
        ac[0] -= 0.01
        # ac[2] -= 0.03
    if ob[1] < 23.4:
        ac[0] += 0.01
        # ac[2] += 0.03
    if ob[2] > 23.6:
        ac[1] -= 0.01
        # ac[3] -= 0.03
    if ob[2] < 23.4:
        ac[1] += 0.01
        # ac[3] += 0.03
    ac[2] = flow_baseline
    ac[3] = flow_baseline
    ac = np.clip(ac, a_min =[-1.0, -1.0, -0.5, -0.5], a_max = [1.0, 1.0, 1.0, 1.0])
    return ac

def manual_controller_v2(ac, ob, flow_baseline=0.0):
    if ob[1] > 26.0:
        ac[0] -= 0.01
        ac[2] += 0.03
    elif ob[1] < 20.0:
        ac[0] += 0.01
        ac[2] += 0.03
    else:
        ac[2] -= 0.01

    if ob[2] > 26.0:
        ac[1] -= 0.01
        ac[3] += 0.03
    elif ob[2] < 20.0:
        ac[1] += 0.01
        ac[3] += 0.03
    else:
        ac[3] -= 0.01

    ac = np.clip(ac, a_min =[-1.0, -1.0, -0.5, -0.5], a_max = [1.0, 1.0, 1.0, 1.0])
    return ac


def traj_segment_generator(pi, env, horizon, stochastic, use_manual_controller_integration=True):
    # Initialize state variables
    t = 0
    if use_manual_controller_integration:
        ac = np.array([-0.7, -0.7, 1.0, 1.0])
        controler_ac = ac.copy()
    else:
        ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    pid_w = PID()
    pid_w.SetPoint = 19.5
    pid_e = PID()
    pid_e.SetPoint = 19.5

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    prevcontact = prevacs.copy()
    # the coeficient of policy actions
    pi_ac_coef = 1
    # set a flag for the time we only use manual controller to fix the situation
    manual_mode_only = False
    # a counter which counts how many timepoints we used the only manual controller for each episode
    manual_mode_only_counter = 0

    while True:
        # we define a mode (manual_only=1, policy and manual =0)
        # in the start we have both of them
        manual_mode=0
        # keep the count of samples which need the manual controller for each iteration
        if manual_mode_only:
            manual_mode_only_counter += 1
            manual_mode = 1        
        # we set the last part of observation
        ob[-1] = manual_mode

        # prevcontact = controler_ac
        # if use the manual controller:
        if use_manual_controller_integration:
            #controler_ac = manual_controller(ac=controler_ac, ob=ob, flow_baseline=1.0)
            pid_w.update(ob[1])
            pid_e.update(ob[2])
            controler_ac = np.clip(np.array([pid_w.output, pid_e.output, 0.7, 0.7]), a_min =[-1.0, -1.0, -0.5, -0.5], a_max = [1.0, 1.0, 1.0, 1.0])

            # add the manual controller action to ob
            ob[6:-1] = controler_ac

            policy_ac, vpred = pi.act(stochastic, ob)
            prevac = policy_ac
            # scale the policy_action using the manual_mode_only
            if manual_mode_only:
                pi_ac_coef = 0
            else:
                pi_ac_coef = 1
            ac = controler_ac + pi_ac_coef*policy_ac
            logger.record_tabular('action_controlerWestSet', controler_ac[0])
            logger.record_tabular('action_policyWestSet', policy_ac[0])
            logger.record_tabular('action_controlerEastSet', controler_ac[1])
            logger.record_tabular('action_policyEastSet', policy_ac[1])
            logger.record_tabular('action_controlerWestFlow', controler_ac[2])
            logger.record_tabular('action_policyWestFlow', policy_ac[2])
            logger.record_tabular('action_controlerEastFlow', controler_ac[3])
            logger.record_tabular('action_policyEastFlow', policy_ac[3])
            logger.record_tabular('action_total_action_West', ac[0])
            logger.record_tabular('action_total_action_East', ac[1])
        else:
            print('not using manual controller')
            prevac = ac
            ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        logger.record_tabular('WestZoneSetPoint', ac[0])
        logger.record_tabular('EastZoneSetPoint', ac[1])
        logger.record_tabular('WestZoneFlow', ac[2])
        logger.record_tabular('EastZoneFlow', ac[3])
        logger.record_tabular('observation_westZoneTemp', ob[1])
        logger.record_tabular('observation_EastZoneTemp', ob[2])

        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            # log the manual controller and set the manual controller counter to zero for new iteration
            logger.record_tabular('Manual_controller_timesteps', manual_mode_only_counter)
            manual_mode_only_counter = 0

        i = t % horizon

        # define the batch
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new        
        if use_manual_controller_integration:
            acs[i] = policy_ac
        else:
            acs[i] = ac
        prevacs[i] = prevac

        # get the next state and evaluate if it is outside the safe range:
            # if outside the range, episode finishes with bad reward, for example -30 and coefpolicy=0 and set flag of manual_mode_only to false
            # if the observation is in the range, coefpolicy=1 and set the flag of manual_mode_only to false
        # manual_mode_only flag controlls if policy actions are important or not, if it is set to False, we dont use policy 
        # action and dont return it
        prevobs = ob
        ob, rew, new, _ = env.step(ac)
        # check if the ob is not inside the range
        entered_bad_zone = False
        entered_good_zone = False
        fake_new = False
        if not new and ((30 < ob[1]) or (16 > ob[1]) or (30 < ob[2]) or (16 > ob[2])):
            print(f'observations are {ob[1]}, {ob[2]}')
            # if we have been in manual mode then just keep the flag true
            if manual_mode_only:
                manual_mode_only = True
            # if we have been using policy before, and just entered the new situation
            else:               
                manual_mode_only = True
                entered_bad_zone=True
                rew += -30.0
                fake_new = True
        
        else:
            # if we are at the manual stage we need to come further inside to have more stable place for agent 
            if manual_mode_only:
                if (28 < ob[1]) or (18 > ob[1]) or (28 < ob[2]) or (18 > ob[2]):
                    # it means the env is not good enough for agent
                    pass
                else:
                    manual_mode_only = False
                    entered_good_zone = True
            
        # add the rewards here
        rews[i] = rew
        cur_ep_ret += rew
        cur_ep_len += 1

        # check if episode actually ended
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()

        # increase time
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn, *,
        timesteps_per_batch, # what to train on
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        callback=None
        ):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        logger.record_tabular('reward_after_seg_mean', np.mean(seg['rew']))
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
