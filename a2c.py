import os.path as osp
import os
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind

from utils import discount_with_dones
from utils import Scheduler, make_path, find_trainable_variables
from utils import cat_entropy, mse
TINY = 1e-8

recon_policies = ['MAReconPolicy']

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs, nplayers,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=4e-4,
            alpha=0.99, epsilon=1e-2, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs*nsteps

        # set i/o placeholders
        use_recon = policy.__name__ in recon_policies # check if it's a model that requires reconstruction.
        recon_shape = [0] if not use_recon else [nbatch*nplayers*512]
        A = tf.placeholder(tf.int32, [nbatch*nplayers])
        ADV = tf.placeholder(tf.float32, [nbatch*nplayers])
        R = tf.placeholder(tf.float32, [nbatch*nplayers])
        LR = tf.placeholder(tf.float32, [])
        RECON = tf.placeholder(tf.float32, recon_shape)

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, nplayers, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, nplayers, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        rec_loss = tf.zeros([1], dtype=tf.float32) # dummy
        if use_recon:
            # recon loss will flow back to rt from RECON.
            # so no need to give it gradient twice.
            rec_loss = tf.reduce_mean(
                            mse(tf.stop_gradient(
                                tf.reshape(train_model.rt, recon_shape)),
                                RECON)
                            )
            loss += rec_loss

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)


        _train = trainer.apply_gradients(grads)
        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)


        def train(obs, states, recons, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            tensors = [pg_loss, rec_loss, vf_loss, entropy, _train, self.summary_op]
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr, RECON:recons}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, recon_loss, value_loss, policy_entropy, _, summary = sess.run(
                    tensors,
                    td_map
            )
            return policy_loss, recon_loss, value_loss, policy_entropy, summary

        def save(save_path, postfix):
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, save_path+"/model"+str(postfix)+".pkl")

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.init_state = step_model.init_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        for grad, var in grads:
            tf.summary.histogram(var.name + '/gradient', grad)
        tf.summary.scalar("learning rate",  LR)
        tf.summary.scalar("pg_loss",  pg_loss)
        tf.summary.scalar("vf_loss",  vf_loss)
        tf.summary.scalar("pred_rewd", tf.reduce_mean(self.train_model.vf))
        tf.summary.scalar("mean_rewd", tf.reduce_mean(R))
        self.summary_op = tf.summary.merge_all()



class Runner(object):

    def __init__(self, env, model, nsteps=16, nstack=12, nplayers=2, gamma=0.99):
        self.env = env
        self.model = model
        self.obs_shape = env.observation_space.shape
        nh, nw, nc = self.obs_shape
        self.nenv = nenv = env.num_envs

        self.nplayers = nplayers
        self.batch_ob_shape = (nenv*nsteps, nplayers, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nplayers, nh, nw, nc*nstack), dtype=np.uint8)

        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.init_state
        self.dones = [[False for x in range(nplayers)] for _ in range(nenv)]

    def update_obs(self, obs):
        nc = self.obs_shape[-1]
        self.obs = np.roll(self.obs, shift=-nc, axis=4)
        self.obs[:, :, :, :, -nc:] = obs

    def run(self):
        mb_obs, mb_states, mb_rewards, mb_recons, mb_actions, mb_values, mb_dones = [], [], [], [], [], [], []
        for n in range(self.nsteps):
            actions, values, states, recons = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_recons.append(recons)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            # the outer loop is per-env done,
            # the inner loop is per-player done
            for i, done in enumerate(dones):
                for j, d in enumerate(done):
                    if d:
                        self.obs[i, j] = self.obs[i, j]*0
            self.update_obs(obs)
            mb_states.append(np.copy(self.states))
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_states = np.asarray(mb_states, dtype=np.float32).swapaxes(1, 0).reshape([self.nenv*self.nsteps, -1])
        mb_recons = np.asarray(mb_recons, dtype=np.float32).swapaxes(1, 0).reshape([self.nenv*self.nsteps*self.nplayers, -1])
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0).swapaxes(2, 1)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0).swapaxes(2, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0).swapaxes(2, 1)
        mb_masks = mb_dones[:, :, :-1]
        mb_dones = mb_dones[:, :, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        for i, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            for j, (r, d, v) in enumerate(zip(rewards, dones, value)):
                r = r.tolist()
                d = d.tolist()
                if d[-1] == 0:
                    r = discount_with_dones(np.asarray(r+[v]), d+[0], self.gamma)[:-1]
                else:
                    r = discount_with_dones(np.asarray(r), d, self.gamma)
                mb_rewards[i, j] = r
            """
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(np.asarray(rewards+[value]), dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(np.asarray(rewards), dones, self.gamma)
            mb_rewards[n] = rewards
            """

        def _flatten(arr):
            return arr.reshape(-1)

        mb_rewards = _flatten(mb_rewards.swapaxes(2,1))
        mb_values = _flatten(mb_values.swapaxes(2,1))
        mb_masks = mb_masks.swapaxes(2,1)
        mb_masks = mb_masks.reshape(mb_masks.shape[0]*mb_masks.shape[1], -1)
        mb_recons = _flatten(mb_recons)
        mb_actions = _flatten(mb_actions)
        return mb_obs, mb_states, mb_recons, mb_rewards, mb_masks, mb_actions, mb_values

def learn(policy, env, seed, checkpoint=0, nsteps=8, nstack=8, nplayers=2,
        total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
        max_grad_norm=0.5, lr=4e-4, lrschedule='linear',
        epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=20, save_interval=20):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
                nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                num_procs=num_procs, nplayers=nplayers,
                ent_coef=ent_coef, vf_coef=vf_coef,
                max_grad_norm=max_grad_norm, lr=lr,
                alpha=alpha, epsilon=epsilon,
                total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    reward_hist = np.array([])
    logs_path = "log/" + policy.__name__
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    if checkpoint != 0:
        print("load from {}".format(checkpoint))
        load_path = logs_path + "/model" + str(checkpoint) + ".pkl"
        model.load(load_path)
    current_ckpt = checkpoint


    for update in range(1, total_timesteps//nbatch+1):
        # Collect batch of samples
        obs, states, recons, rewards, masks, actions, values = runner.run()
        # Train on batch
        policy_loss, recon_loss, value_loss, policy_entropy, summary = model.train(obs, states, recons, rewards, masks, actions, values)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        current_ckpt += 1

        # reshape to obtain the score of each agent
        if update % log_interval == 0 or update == 1:
            summary_writer.add_summary(summary, update)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("recon_loss", float(recon_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained variance", float(ev))
            rewards = rewards.reshape(-1, 2)
            for i in range(nplayers):
                logger.record_tabular("max_reward_%d" % i, float(np.max(rewards[:, i])))
                logger.record_tabular("mean_reward_%d (over 100 updates)" % i, float(np.mean(rewards[:, i])))
            logger.dump_tabular()
            if update % (log_interval * save_interval) == 0:
                model.save(logs_path, current_ckpt)
    env.close()
    model.save(logs_path, current_ckpt)
    return current_ckpt

if __name__ == '__main__':
    main()
