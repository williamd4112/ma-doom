import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_state_ops
from utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample, check_shape, lnmem, nmap
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U
import gym

class MANMapPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack,
            nplayers, map_size=[15, 15, 32], reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nplayers, nh, nw, nc*nstack)
        nact = ac_space.n

        X = tf.placeholder(tf.uint8, ob_shape, name='X')
        MAP = tf.placeholder(tf.float32, [nbatch,] + map_size, name='MEM')
        C = tf.placeholder(tf.int32, [nbatch, nplayers, 2])

        pis = []
        vfs = []

        with tf.variable_scope("mem-var"):
            m = tf.get_variable("mem-var-%d" % nbatch, shape=MAP.get_shape(), trainable=False)

        with tf.variable_scope("model", reuse=reuse):
            # tuck observation from all players at once
            x = tf.reshape(tf.cast(X, tf.float32)/255., [nbatch*nplayers, nh, nw, nc*nstack])
            m = tf.assign(m, MAP)

            h1 = conv( x, 'conv1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h1, 'conv2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'conv3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))

            # shared memory:
            # instead of time-sequence, each rnn cell here
            # is responsible for "one player"

            # mem_size : half for key, half for value
            xs = batch_to_seq(h4, nenv*nsteps, nplayers)
            crds = batch_to_seq(C, nenv*nsteps, nplayers)
            context, rs, ws, map_new = nmap(xs, m, crds, 'mem', nplayers, n=map_size[1], feat=map_size[-1])
            h4 = tf.reshape(h4, [nbatch, nplayers, -1])

            context = tf.reshape(context, [nbatch, nplayers, -1])
            rs = tf.reshape(rs, [nbatch, nplayers, -1])
            ws = tf.reshape(ws, [nbatch, nplayers, -1])

            # compute pi, vaule for each agents
            print(ws)

            _reuse = False
            for i in range(nplayers):
                h5 = fc(tf.concat([context[:, i], rs[:, i], ws[:, i]], axis=1), 'fc-pi', nh=512, init_scale=np.sqrt(2), reuse=_reuse)
                pi = fc(h5, 'pi', nact, act=tf.identity, reuse=_reuse)
                vf = fc(h5, 'v', 1, act=tf.identity, reuse=_reuse)
                pis.append(pi)
                vfs.append(vf)
                _reuse = True
            pi = tf.reshape(tf.concat(pis, axis=1), [nbatch*nplayers, -1])
            vf = tf.reshape(tf.concat(vfs, axis=1), [nbatch*nplayers, -1])

        v0 = vf
        a0 = sample(pi)

        self.init_state = np.zeros([nbatch,]+map_size, dtype=np.float32)

        def step(ob, maps, coords):
            a, v, m = sess.run([a0, v0, map_new], {X:ob, MAP:maps, C:coords})
            a = [a[i:i+nplayers] for i in range(0, len(a), nplayers)]
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return a, v, m, [] # dummy recon

        def value(ob, state, mask):
            v = sess.run(v0, {X:ob, S:state, M:mask})
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return v

        self.X = X
        self.MAP = MAP # to meet the a2c.py protocol.
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
            pi = tf.reshape(tf.concat(pis, axis=1), [nbatch*nplayers, -1])
            vf = tf.reshape(tf.concat(vfs, axis=1), [nbatch*nplayers, -1])

        v0 = vf
        a0 = sample(pi)

        self.init_state = np.zeros((nbatch, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            a = [a[i:i+nplayers] for i in range(0, len(a), nplayers)]
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return a, v, s, [] # dummy recon

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
            h = conv(x, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            h5 = fc(h4, 'fc2', nh=512, init_scale=np.sqrt(2))
            h6 = fc(h5, 'fc3', nh=256, init_scale=np.sqrt(2)) # to give compatible network size
            h6 = tf.reshape(h6, [nbatch, nplayers, -1])

            _reuse = False
            for i in range(nplayers):
                pi = fc(h6[:,i], 'pi', nact, act=tf.identity, reuse=_reuse)
                vf = fc(h6[:,i], 'v', 1, act=tf.identity, reuse=_reuse)
                pis.append(pi)
                vfs.append(vf)
                _reuse = True
            pi = tf.reshape(tf.concat(pis, axis=1), [nbatch*nplayers, -1])
            vf = tf.reshape(tf.concat(vfs, axis=1), [nbatch*nplayers, -1])

        v0 = vf
        a0 = sample(pi)

        self.init_state = []


        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            a = [a[i:i+nplayers] for i in range(0, len(a), nplayers)]
            v = [v[i:i+nplayers] for i in range(0, len(v), nplayers)]
            return a, v, [], [] #dummy state and recon

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
        model = MANMapPolicy(sess, spaces.Box(low=0, high=255, shape=(84, 84, 3)), spaces.Discrete(3), 8, 4, 4, 2)
        tf.global_variables_initializer().run(session=sess)

        rets = (model.step(np.random.rand(32, 2, 84, 84, 12), np.random.rand(32,15,15,32), np.random.randint(0, 2, (32, 2, 2))))
        #print(rets)
