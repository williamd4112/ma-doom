import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, check_shape
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U
import gym

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class MACnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv,
                    nsteps, nstack, nplayers, reuse=False, merge=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nplayers, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs

        # Communication phase
        encs = []
        h4s = []
        for i in range(nplayers):
            x = X[:, i, ::]
            with tf.variable_scope("model_%d" % i, reuse=reuse):
                h = conv(tf.cast(x, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
                h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
                h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
                h3 = conv_to_fc(h3)
                h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), act=lambda x:x)
                # Encode visual features to communication message (512 -> 128)
                enc = fc(h4, 'fc1-enc', nh=128, act=tf.nn.relu)
            encs.append(enc)
            h4s.append(h4)
        # Policy phase
        pis = []
        vfs = []
        decs = []
        for i in range(nplayers):
            comm = [ e for j, e in enumerate(encs) if i != j]
            comm = tf.add_n(comm) / (nplayers - 1)
            h4 = h4s[i]
            with tf.variable_scope("model_%d" % i, reuse=reuse):
                # Decode communication message to features (128 -> 512)
                dec = fc(comm, 'fc1-dec', nh=512, act=tf.nn.relu)
                dec_ = tf.stop_gradient(tf.identity(dec, name='dec_-%d' % i))
                att = fc(dec_, 'fc2-att', nh=512, act=tf.nn.relu)
                # Policy and Value function
                if merge:
                    feats = tf.multiply(h4, att)
                else:
                    feats = tf.concat([h4, att], axis=1)
                pi = fc(feats, 'pi', nact, act=lambda x:x)
                vf = fc(feats, 'v', 1, act=lambda x:x)
            pis.append(pi)
            vfs.append(vf)
            decs.append(tf.identity(dec, name='dec-%d' % i))

        v0 = tf.tuple([ vf[:, 0] for vf in vfs ])
        a0 = tf.tuple([ sample(pi) for pi in pis ])

        self.initial_state = [] #not statefu

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return np.stack(a, axis=1), np.stack(v, axis=1), [] #dummy state

        def value(ob, *_args, **_kwargs):
            v = sess.run(v0, {X:ob})
            return np.stack(v, axis=1)

        # X is [nbatch, nplayer, h, w, c*hist_len]
        self.X = X
        # pi is [[nbatch, nact]] * nplayers
        self.pi = pis
        # vf is [[nbatch, 1]] * nplayers
        self.vf = vfs
        # step return is [(a)[nbatch,] * nplayers, (v)[nbatch,] * nplayers, []]
        self.step = step
        # value return is [(v)[nbatch,] * nplayers, []]
        self.value = value
        self.orig = [h4s[1], h4s[0]]
        self.dec = decs


if __name__ == '__main__':
    from gym import error, spaces
    X = tf.placeholder(tf.uint8, (32, 2, 84, 84, 12))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = MACnnPolicy(sess, spaces.Box(low=0, high=255, shape=(84, 84, 3)), spaces.Discrete(3), 8, 4, 4, 2)
        tf.global_variables_initializer().run(session=sess)
        print(model.pi)
        print(model.vf)

        rets = (model.step(np.random.rand(32, 2, 84, 84, 12)))
        #for a in rets[0]:
        #    print(a)
        #for v in rets[1]:
        #    print(v)
        print(rets[0].shape)
        print(rets[1].shape)
        rets = (model.value(np.random.rand(32, 2, 84, 84, 12)))
        print(rets.shape)
