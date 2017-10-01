import numpy as np
import tensorflow as tf
from utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, check_shape, lnmem
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U
import gym

class MACommSepCriticPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack,
            nplayers, nlstm=256, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nplayers, nh, nw, nc*nstack)
        nact = ac_space.n

        X = tf.placeholder(tf.uint8, ob_shape, name='X')
        M = tf.placeholder(tf.float32, [nbatch, nplayers], name='M')
        S = tf.placeholder(tf.float32, [nbatch, nlstm*2], name='S')

        pis = []
        vfs = []

        with tf.variable_scope("model", reuse=reuse):
            # tuck observation from all players at once
            x = tf.reshape(tf.cast(X, tf.float32)/255., [nbatch*nplayers, nh, nw, nc*nstack])
            h1 = conv( x, 'conv1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h1, 'conv2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'conv3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))

            # shared memory:
            # instead of time-sequence, each rnn cell here
            # is responsible for "one player"
            xs = batch_to_seq(h4, nenv*nsteps, nplayers)
            ms = tf.expand_dims(M, axis=1)
            #ms = batch_to_seq( M, nenv*nsteps, nplayers)

            mem, snew = lnmem(xs, ms, S, 'lstm1', nh=nlstm)
            mem = tf.reshape(mem, [nbatch, nlstm*2])
            mem = fc(mem, 'fcmem', nh=256, init_scale=np.sqrt(2))
            #tf.summary.histogram('rnn_activation', mem)
            h4 = tf.reshape(h4, [nbatch, nplayers, -1])

            _reuse = False
            for i in range(nplayers):
                # shared critic, separate from policy module
                vf = fc(mem, 'vf', nh=1, act=tf.identity, reuse=_reuse)
                _vf = tf.stop_gradient(vf)

                h5 = fc(tf.concat([_vf, h4[:,i]], axis=1), 'fc-pi', nh=512, init_scale=np.sqrt(2), reuse=_reuse)
                pi = fc(h5, 'pi', nact, act=tf.identity, reuse=_reuse)
                pis.append(pi)
                vfs.append(vf)
                _reuse = True
            pi = tf.concat(pis, axis=0)
            vf = tf.concat(vfs, axis=0)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.init_state = np.zeros((nbatch, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            a = [a[i:i+nplayers] for i in range(0, len(a), nplayers)]
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return a, v, s

        def value(ob, state, mask):
            v = sess.run(v0, {X:ob, S:state, M:mask})
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return v

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class MACommPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack,
            nplayers, nlstm=256, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nplayers, nh, nw, nc*nstack)
        nact = ac_space.n

        X = tf.placeholder(tf.uint8, ob_shape, name='X')
        M = tf.placeholder(tf.float32, [nbatch, nplayers], name='M')
        S = tf.placeholder(tf.float32, [nbatch, nlstm*2], name='S')

        pis = []
        vfs = []

        with tf.variable_scope("model", reuse=reuse):
            # tuck observation from all players at once
            x = tf.reshape(tf.cast(X, tf.float32)/255., [nbatch*nplayers, nh, nw, nc*nstack])
            h1 = conv( x, 'conv1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h1, 'conv2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'conv3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))

            # shared memory:
            # instead of time-sequence, each rnn cell here
            # is responsible for "one player"
            xs = batch_to_seq(h4, nenv*nsteps, nplayers)
            #ms = tf.expand_dims(M, axis=1)
            ms = batch_to_seq( M, nenv*nsteps, nplayers)

            mem, snew = lnmem(xs, ms, S, 'lstm1', nh=nlstm)
            mem = tf.reshape(mem, [nbatch, nlstm*2])
            mem = fc(mem, 'fcmem', nh=256, init_scale=np.sqrt(2))
            #tf.summary.histogram('rnn_activation', mem)
            h4 = tf.reshape(h4, [nbatch, nplayers, -1])

            # compute pi, vaule for each agents

            _reuse = False
            for i in range(nplayers):
                h5 = fc(tf.concat([mem, h4[:,i]], axis=1), 'fc-pi', nh=512, init_scale=np.sqrt(2), reuse=_reuse)
                pi = fc(h5, 'pi', nact, act=tf.identity, reuse=_reuse)
                vf = fc(h5, 'v', 1, act=tf.identity, reuse=_reuse)
                pis.append(pi)
                vfs.append(vf)
                _reuse = True
            pi = tf.concat(pis, axis=0)
            vf = tf.concat(vfs, axis=0)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.init_state = np.zeros((nbatch, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            a = [a[i:i+nplayers] for i in range(0, len(a), nplayers)]
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return a, v, s

        def value(ob, state, mask):
            v = sess.run(v0, {X:ob, S:state, M:mask})
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return v

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class MACnnSepPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv,
                    nsteps, nstack, nplayers, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nplayers, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs

        pis = []
        vfs = []

        _reuse = reuse
        for i in range(nplayers):
            with tf.variable_scope("model", reuse=reuse):
                x = tf.cast(X[:, i], tf.float32)/255.
                h = conv(tf.cast(x, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), reuse=_reuse)
                h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), reuse=_reuse)
                h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), reuse=_reuse)
                h3 = conv_to_fc(h3)
                h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2), reuse=_reuse)
                h5 = fc(h4, 'fc2', nh=512, init_scale=np.sqrt(2), reuse=_reuse)

                pi = fc(h5, 'pi', nact, act=tf.identity, reuse=_reuse)
                vf = fc(h5, 'v', 1, act=tf.identity, reuse=_reuse)
            pis.append(pi)
            vfs.append(vf)
            _reuse = True
        pi = tf.concat(pis, axis=0)
        vf = tf.concat(vfs, axis=0)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.init_state = []


        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            a = [a[i:i+nplayers] for i in range(0, len(a), nplayers)]
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            v = sess.run(v0, {X:ob})
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return v

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value



class MACnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv,
                    nsteps, nstack, nplayers, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nplayers, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs

        pis = []
        vfs = []

        with tf.variable_scope("model", reuse=reuse):
            x = tf.reshape(tf.cast(X, tf.float32)/255., [nbatch*nplayers, nh, nw, nc*nstack])
            h = conv(tf.cast(x, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            h5 = fc(h4, 'fc2', nh=512, init_scale=np.sqrt(2))
            h5 = tf.reshape(h5, [nbatch, nplayers, -1])

            _reuse = False
            for i in range(nplayers):
                pi = fc(h5[:,i], 'pi', nact, act=tf.identity, reuse=_reuse)
                vf = fc(h5[:,i], 'v', 1, act=tf.identity, reuse=_reuse)
                pis.append(pi)
                vfs.append(vf)
                _reuse = True
            pi = tf.concat(pis, axis=0)
            vf = tf.concat(vfs, axis=0)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.init_state = []


        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            a = [a[i:i+nplayers] for i in range(0, len(a), nplayers)]
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            v = sess.run(v0, {X:ob})
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return v

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


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
