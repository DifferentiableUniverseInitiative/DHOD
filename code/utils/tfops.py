import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
import warnings
import tensorflow_probability as tfp
import numpy as np
tfb = tfp.bijectors



def dynamic_deconv_op3d(x, W, strides=[1,2,2,2,1], padding='SAME'):

    filter_size = tf.shape(W)#list(map(int, W.get_shape()))
    xs = tf.shape(x)
    target_shape = tf.shape(x)
    #channelscale = int(int(W.get_shape()[-1])/int(x.get_shape()[-1]))
    if padding == 'SAME': 
        shapescaling = tf.constant([1, strides[1], strides[2], strides[3], 1])
        print(shapescaling)
        target_shape = target_shape*shapescaling
    if padding == 'VALID': 
        shapescaling = tf.constant([1, strides[1], strides[2], strides[3], 1])
        shapeoffset =  tf.constant([0, filter_size[0]-1, filter_size[1]-1, filter_size[2]-1, 1])
        #This doesnt seem to be working. Figure out at some point
        target_shape = target_shape*shapescaling
        target_shape = target_shape+shapeoffset
    return tf.nn.conv3d_transpose(x, W, target_shape, strides, padding)


def dynamic_deconv3d(name, x, shape, strides=[1,2,2,2,1], activation=tf.nn.leaky_relu):
    in_dim = x.get_shape()[-1]
    W_shape = [shape[0], shape[1], shape[2], in_dim, shape[3]]
    W = tf.get_variable(name+'_W', W_shape, tf.float32, tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name+'_b', W_shape[3], tf.float32, tf.zeros_initializer)
    conv = dynamic_deconv_op3d(x, W, strides=strides)
    return activation(tf.add(conv, b))



###Following spectral normed conv3d has been contributed by Francois Lanusse

NO_OPS = 'NO_OPS'
def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False, normwt=0.7):
  # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])

    if u is None:
        u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1

    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32),
                   u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
      )

    if update_collection is None:
        warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                  '. Please consider using a update collection instead.')
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        scaling = tf.minimum(1., normwt/sigma)
        #scaling = normwt/sigma
        W_bar = W_reshaped *scaling
        with tf.control_dependencies([u.assign(u_final)]):
            W_bar = tf.reshape(W_bar, W_shape)
    else:
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)
        # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
        # has already been collected on the first call.
        if update_collection != NO_OPS:
            tf.add_to_collection(update_collection, u.assign(u_final))
    if with_sigma:
        return W_bar, sigma
    else:
        return W_bar


@add_arg_scope
def scope_has_variables(scope):
    return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0

@add_arg_scope
def specnormconv3d(input_, output_dim,
           kernel_size=3, stride=1, stddev=None,
           name="conv3d", spectral_normed=True, update_collection=None, with_w=False,
                   padding="SAME", reuse=tf.AUTO_REUSE, num_iters=1, normwt=1):

    k_h, k_w, k_z = [kernel_size]*3
    d_h, d_w, d_z = [stride]*3
    # Glorot intialization
    # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
    fan_in = k_h * k_w * k_z * input_.get_shape().as_list()[-1]
    fan_out = k_h * k_w * k_z * output_dim
    if stddev is None:
        stddev = tf.sqrt(2. / (fan_in))

    with tf.variable_scope(name) as scope:
        if scope_has_variables(scope):
            scope.reuse_variables()
        w = tf.get_variable("w", [k_h, k_w, k_z, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if spectral_normed:
            conv = tf.nn.conv3d(input_, spectral_normed_weight(w, update_collection=update_collection, num_iters=num_iters, normwt=normwt),
                              strides=[1, d_h, d_w, d_z, 1], padding=padding)
        else:
            conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, d_z, 1], padding=padding)

        biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if with_w:
            return conv, w, biases
        else:
            return conv



@add_arg_scope
def specnormconv2d(input_, output_dim,
           kernel_size=3, stride=1, stddev=None,
           name="conv2d", spectral_normed=True, update_collection=None, with_w=False,
                   padding="SAME", reuse=tf.AUTO_REUSE, num_iters=1, normwt=1.):

    k_h, k_w = [kernel_size]*2
    d_h, d_w = [stride]*2
    # Glorot intialization
    # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
    fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
    fan_out = k_h * k_w * output_dim
    if stddev is None:
        stddev = tf.sqrt(2. / (fan_in))

    with tf.variable_scope(name) as scope:
        if scope_has_variables(scope):
            scope.reuse_variables()
        w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if spectral_normed:
            conv = tf.nn.conv2d(input_, spectral_normed_weight(w, update_collection=update_collection,
                                                               num_iters=num_iters, normwt=normwt),
                              strides=[1, d_h, d_w, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if with_w:
            return conv, w, biases
        else:
            return conv


@add_arg_scope
def specnormconv1d(input_, output_dim,
           kernel_size=3, stride=1, stddev=None,
           name="conv1d", spectral_normed=True, update_collection=None, with_w=False,
                   padding="SAME", reuse=tf.AUTO_REUSE, num_iters=1):

    k_h = kernel_size
    d_h = stride
    # Glorot intialization
    # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
    fan_in = k_h * input_.get_shape().as_list()[-1]
    fan_out = k_h * output_dim
    if stddev is None:
        stddev = tf.sqrt(2. / (fan_in))

    with tf.variable_scope(name) as scope:
        if scope_has_variables(scope):
            scope.reuse_variables()
        w = tf.get_variable("w", [k_h, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        #w = tf.get_variable("w", [k_h, input_.get_shape()[-1], output_dim],
        #                    initializer=tf.glorot_uniform_initializer())
        if spectral_normed:
            conv = tf.nn.conv1d(input_, spectral_normed_weight(w, update_collection=update_collection,
                                                               num_iters=num_iters),
                              stride=d_h, padding=padding)
        else:
            conv = tf.nn.conv1d(input_, w, stride=d_h, padding=padding)

        biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if with_w:
            return conv, w, biases
        else:
            return conv



###Following have been taken from https://github.com/openai/glow/blob/master/tfops.py
###and modified to 3D. In addition, some variable names have been modified
###and assumptions on shape of input being constant relaxed

def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))

# Invertible 1x1 conv
@add_arg_scope
def invertible_1x1_conv(name, z, reverse=False):

    if True:  # Set to "False" to use the LU-decomposed version

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            shape = int_shape(z)
            w_shape = [shape[-1], shape[-1]]

            # Sample a random orthogonal matrix:
            w_init = np.linalg.qr(np.random.randn(
                *w_shape))[0].astype('float32')

            w = tf.get_variable("W", dtype=tf.float32, initializer=w_init)
            # dlogdet = tf.linalg.LinearOperator(w).log_abs_determinant() * shape[1]*shape[2]
            logdet = tf.cast(tf.log(abs(tf.matrix_determinant(
                tf.cast(w, 'float64')))), 'float32') * shape[1]*shape[2]*shape[3]
            if not reverse:

                _w = tf.reshape(w, [1, 1, 1] + w_shape)
                z = tf.nn.conv3d(z, _w, [1, 1, 1, 1, 1],
                                 'SAME', data_format='NDHWC')
                return z, logdet
            else:

                _w = tf.matrix_inverse(w)
                _w = tf.reshape(_w, [1, 1, 1]+w_shape)
                z = tf.nn.conv3d(z, _w, [1, 1, 1, 1, 1],
                                 'SAME', data_format='NDHWC')
                return z, -logdet

    else:

        # LU-decomposed version
        shape = int_shape(z)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            dtype = 'float64'

            # Random orthogonal matrix:
            import scipy
            np_w = scipy.linalg.qr(np.random.randn(shape[-1], shape[-1]))[
                0].astype('float32')

            np_p, np_l, np_u = scipy.linalg.lu(np_w)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)

            p = tf.get_variable("P", initializer=np_p, trainable=False)
            l = tf.get_variable("L", initializer=np_l)
            sign_s = tf.get_variable(
                "sign_S", initializer=np_sign_s, trainable=False)
            log_s = tf.get_variable("log_S", initializer=np_log_s)
            # S = tf.get_variable("S", initializer=np_s)
            u = tf.get_variable("U", initializer=np_u)

            p = tf.cast(p, dtype)
            l = tf.cast(l, dtype)
            sign_s = tf.cast(sign_s, dtype)
            log_s = tf.cast(log_s, dtype)
            u = tf.cast(u, dtype)

            w_shape = [shape[-1], shape[-1]]

            l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
            l = l * l_mask + tf.eye(*w_shape, dtype=dtype)
            u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
            w = tf.matmul(p, tf.matmul(l, u))

            if True:
                u_inv = tf.matrix_inverse(u)
                l_inv = tf.matrix_inverse(l)
                p_inv = tf.matrix_inverse(p)
                w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
            else:
                w_inv = tf.matrix_inverse(w)

            w = tf.cast(w, tf.float32)
            w_inv = tf.cast(w_inv, tf.float32)
            log_s = tf.cast(log_s, tf.float32)

            if not reverse:

                w = tf.reshape(w, [1, 1, 1] + w_shape)
                z = tf.nn.conv3d(z, w, [1, 1, 1, 1, 1],
                                 'SAME', data_format='NDHWC')
                logdet = tf.reduce_sum(log_s) * (shape[1]*shape[2]*shape[3])

                return z, logdet
            else:

                w_inv = tf.reshape(w_inv, [1, 1, 1]+w_shape)
                z = tf.nn.conv3d(
                    z, w_inv, [1, 1, 1, 1, 1], 'SAME', data_format='NDHWC')
                logdet = -tf.reduce_sum(log_s) * (shape[1]*shape[2]*shape[3])

                return z, logdet


@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False, trainable=True):
    w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w
    return w

@add_arg_scope
def actnorm3d(x, scale=1., logscale_factor=3., batch_variance=False,
            reverse=False, init=False, is_training=True, scope='actnorm'):
    """
    Borrowed from https://github.com/openai/glow/blob/master/tfops.py
    """
    name = scope
    if arg_scope([get_variable_ddi], trainable=is_training):
        if not reverse:
            x = actnorm_center3d(name+"_center", x, reverse)
            x, logdet = actnorm_scale3d(name+"_scale", x, scale,
                              logscale_factor, batch_variance, reverse, init)
        else:
            x, logdet = actnorm_scale3d(name + "_scale", x, scale,
                              logscale_factor, batch_variance, reverse, init)

            x = actnorm_center3d(name+"_center", x, reverse)
        return x, logdet

@add_arg_scope
def actnorm_center3d(name, x, reverse=False):
    shape = x.get_shape()
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        assert len(shape) == 5
        x_mean = tf.reduce_mean(x, [0, 1, 2, 3], keepdims=True)
        b = get_variable_ddi(
#             "b", (1, 1, 1, 1, int_shape(x)[4]), initial_value=-x_mean)
            "b", (1, 1, 1, 1, shape[4]), initial_value=-x_mean)

        if not reverse:
            x += b
        else:
            x -= b

        return x


@add_arg_scope
def actnorm_scale3d(name, x, scale=1., logscale_factor=3.,
                  batch_variance=False, reverse=False, init=False, trainable=True):
    shape = x.get_shape()
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE), arg_scope([get_variable_ddi], trainable=trainable):
        assert len(shape) == 5
        x_var = tf.reduce_mean(x**2, [0, 1, 2, 3], keepdims=True)
#         logdet_factor = int(shape[1])*int(shape[2])*int(shape[3])
        logdet_factor = int(shape[1])*int(shape[2])*int(shape[3])
#         _shape = (1, 1, 1, 1, int_shape(x)[4])
        _shape = (1, 1, 1, 1, shape[4])

        if batch_variance:
            x_var = tf.reduce_mean(x**2, keepdims=True)

        if True:
            logs = get_variable_ddi("logs", _shape, initial_value=tf.log(
                scale/(tf.sqrt(x_var)+1e-6))/logscale_factor)*logscale_factor
            if not reverse:
                x = x * tf.exp(logs)
            else:
                x = x * tf.exp(-logs)
        else:
            # Alternative, doesn't seem to do significantly worse or better than the logarithmic version above
            s = get_variable_ddi("s", _shape, initial_value=scale /
                                 (tf.sqrt(x_var) + 1e-6) / logscale_factor)*logscale_factor
            logs = tf.log(tf.abs(s))
            if not reverse:
                x *= s
            else:
                x /= s

        dlogdet = tf.reduce_sum(logs) * logdet_factor
        if reverse:
            dlogdet *= -1
        return x, dlogdet


#Squeeze operation for invertible downsampling in 3D
#
#
class Squeeze3d(tfb.Reshape):
    """
    Borrowed from https://github.com/openai/glow/blob/master/tfops.py
    """
    def __init__(self,
                 event_shape_in,
                 factor=2,
                 is_constant_jacobian=True,
                 validate_args=False,
                 name=None):

        assert factor >= 1
        name = name or "squeeze"
        self.factor = factor
        event_shape_out = 1*event_shape_in
        event_shape_out[0] //=2
        event_shape_out[1] //=2
        event_shape_out[2] //=2
        event_shape_out[3] *=8
        self.event_shape_out = event_shape_out

        super(Squeeze3d, self).__init__(
            event_shape_out=event_shape_out,
            event_shape_in=event_shape_in,
        validate_args=validate_args,
        name=name)

    def _forward(self, x):
        if self.factor == 1:
            return x
        factor = self.factor

        shape = tf.shape(x)
        height = shape[1]
        width = shape[2]
        length = shape[3]
        n_channels = x.get_shape()[4]

#         print(height, width, length, n_channels )
#         assert height % factor == 0 and width % factor == 0 and length % factor == 0
        x = tf.reshape(x, [-1, height//factor, factor,
                           width//factor, factor, length//factor, factor, n_channels])
        x = tf.transpose(x, [0, 1, 3, 5, 7, 2, 4, 6])
        x = tf.reshape(x, [-1, height//factor, width//factor,
                               length//factor, n_channels*factor**3])
        return x

    def _inverse(self, x):
        if self.factor == 1:
            return x
        factor = self.factor

        shape = tf.shape(x)
        height = shape[1]
        width = shape[2]
        length = shape[3]
        n_channels = int(x.get_shape()[4])

#         print(height, width, length, n_channels )
        assert n_channels >= 8 and n_channels % 8 == 0
        x = tf.reshape(
            x, [-1, height, width, length, int(n_channels/factor**3), factor, factor, factor])
        x = tf.transpose(x, [0, 1, 5, 2, 6, 3, 7, 4])
        x = tf.reshape(x, (-1, height*factor,
                           width*factor, height*factor, int(n_channels/factor**3)))
        return x


def f_net(name, h, width, n_out=None):
    n_out = n_out or int(h.get_shape()[4])
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        h = tf.layers.conv3d(h, width, 3, padding='SAME', activation=tf.nn.relu)
        h = tf.layers.conv3d(h, n_out, 3, padding='SAME', activation=None)
    return h

def f_net3d(name, h, width, n_out=None):

    k_h, k_w, k_z = [3]*3
    fan_in = k_h * k_w * k_z * h.get_shape().as_list()[-1]
    stddev = tf.sqrt(2. / (fan_in))
    n_out = n_out or int(h.get_shape()[4])

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#         if scope_has_variables(scope):
#             scope.reuse_variables()
        w1 = tf.get_variable("w1", [3, 3, 3, h.get_shape()[-1], width],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        w2 = tf.get_variable("w2", [3, 3, 3, width, width],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        wout = tf.get_variable("wout", [3, 3, 3, width, n_out],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        h = tf.nn.relu(tf.nn.conv3d(h, w1, strides=[1, 1, 1, 1, 1], padding='SAME'))
        h = tf.nn.relu(tf.nn.conv3d(h, w2, strides=[1, 1, 1, 1, 1], padding='SAME'))
        h = tf.nn.conv3d(h, wout, strides=[1, 1, 1, 1, 1], padding='SAME')
    return h
