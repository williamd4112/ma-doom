import os.path as osp
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

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs, nplayers,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-2, total_timesteps=int(80e6), lrschedule='linear', no_recon=False, merge=False):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch, nplayers])
        ADV = tf.placeholder(tf.float32, [nbatch, nplayers])
        R = tf.placeholder(tf.float32, [nbatch, nplayers])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, nplayers, reuse=False, merge=merge)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, nplayers, reuse=True, merge=merge)

        # Calculate multiplayer loss
        pg_losses = []
        vf_losses = []
        recon_losses = []
        entropies = []
        losses = []
        for i in range(nplayers):
            neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi[i], labels=A[:, i])
            pg_loss = tf.reduce_mean(ADV[:, i] * neglogpac)
            vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf[i]), R[:, i]))
            recon_loss = tf.reduce_mean(mse(tf.squeeze(train_model.dec[i]),
                                            tf.squeeze(train_model.orig[i])))
            entropy = tf.reduce_mean(cat_entropy(train_model.pi[i]))
            if not no_recon:
                loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef + recon_loss
            else:
                loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef
            pg_losses.append(pg_loss)
            vf_losses.append(vf_loss)
            recon_losses.append(recon_loss)
            entropies.append(entropy)
            losses.append(loss)
        pg_loss = tf.tuple(pg_losses)
        vf_loss = tf.tuple(vf_losses)
        entropy = tf.tuple(entropies)
        recon_loss = tf.tuple(recon_losses)
        loss = tf.add_n(losses)

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, reconstruct_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, recon_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, reconstruct_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, save_path)

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
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, nstack=4, nplayers=2, gamma=0.99):
        self.env = env
        self.model = model
        self.nplayers = nplayers

        self.obs_shape = env.observation_space.shape
        nh, nw, nc = self.obs_shape

        nenv = env.num_envs

        self.batch_ob_shape = (nenv*nsteps, nplayers, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nplayers, nh, nw, nc*nstack), dtype=np.uint8)

        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def update_obs(self, obs):
        nc = self.obs_shape[-1]
        self.obs = np.roll(self.obs, shift=-nc, axis=4)
        self.obs[:, :, :, :, -nc:] = obs

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(np.asarray(rewards+[value]), dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(np.asarray(rewards), dones, self.gamma)
            mb_rewards[n] = rewards

        def _flatten(arr):
            shape = arr.shape
            return arr.reshape([shape[0]*shape[1], -1])

        mb_rewards = _flatten(mb_rewards)
        mb_actions = _flatten(mb_actions)
        mb_values = _flatten(mb_values)
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

def learn(policy, env, seed, nsteps=5, nstack=4, nplayers=2,
        total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
        max_grad_norm=0.5, lr=7e-4, lrschedule='linear',
        epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=4, no_recon=False, merge=False):
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
                total_timesteps=total_timesteps, lrschedule=lrschedule,
                no_recon=no_recon,
                merge=merge)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        # Collect batch of samples
        obs, states, rewards, masks, actions, values = runner.run()
        # Train on batch
        policy_loss, value_loss, reconstruct_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            for i in range(nplayers):
                ev = explained_variance(values[:, i], rewards[:, i])
                logger.record_tabular("policy_entropy_%d" % i, float(policy_entropy[i]))
                logger.record_tabular("value_loss_%d" % i, float(value_loss[i]))
                logger.record_tabular("reconstruct_loss_%d" % i, float(reconstruct_loss[i]))
                logger.record_tabular("explained_variance_%d" % i, float(ev))
                logger.record_tabular("mean_reward_%d" % i, float(np.mean(rewards[:, i])))
                logger.record_tabular("max_reward_%d" % i, float(np.max(rewards[:, i])))
            logger.dump_tabular()
    env.close()

if __name__ == '__main__':
    main()
