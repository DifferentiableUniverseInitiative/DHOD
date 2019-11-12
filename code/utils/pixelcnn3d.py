import tensorflow as tf
import numpy as np

def get_weights3d(shape, name, orientation, mask_mode='noblind', mask=None):
    weights_initializer = tf.contrib.layers.xavier_initializer()
#     weights_initializer = tf.ones_initializer()
    W = tf.get_variable(name, shape, tf.float32, weights_initializer)

    '''
        Use of masking to hide subsequent pixel values 
    '''
    if mask:
        filter_mid_x = shape[0]//2
        filter_mid_y = shape[1]//2
        filter_mid_z = shape[2]//2
        
        mask_filter = np.ones(shape, dtype=np.float32)
        if mask_mode == 'noblind':
                
            if orientation == 0:
                if mask == 'a':
                    mask_filter[filter_mid_x:, :, :, :, :] = 0.0
                else:
                    mask_filter[filter_mid_x+1:, :, :, :, :] = 0.0

            elif orientation == 1 :
                mask_filter[filter_mid_x+1:, :, :, :, :] = 0.0
                if mask == 'a':
                    mask_filter[filter_mid_x:, filter_mid_y:, :, :, :] = 0.0
                else:
                    mask_filter[filter_mid_x:, filter_mid_y+1:, :, :, :] = 0.0

            elif orientation == 2:
                mask_filter[filter_mid_x+1:, :, :, :, :] = 0.0
                mask_filter[filter_mid_x:, filter_mid_y+1:, :, :, :] = 0.0
                mask_filter[filter_mid_x:, filter_mid_y:, filter_mid_z+1:, :, :] = 0.0

            if mask == 'a':
                # Center must be zero in first layer
                mask_filter[filter_mid_x, filter_mid_y, filter_mid_z, :, :] = 0.0

        else:
            mask_filter[filter_mid_x, filter_mid_y:, filter_mid_z+1:, :, :] = 0.
            mask_filter[filter_mid_x, filter_mid_y+1:, :, :, :] = 0.
            mask_filter[filter_mid_x+1:, :, :, :, :] = 0.

            if mask == 'a':
                mask_filter[filter_mid_x, filter_mid_y, filter_mid_z, :, :] = 0.
                mask_filter[filter_mid_x:, filter_mid_y:, :, :, :] = 0.0
                mask_filter[filter_mid_x:, :, :, :, :] = 0.0
                
        W = W*mask_filter 
#         W *= mask_filter
#         W.assign(W * mask_filter)
    return W


def get_bias(shape, name):
    return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer)

def conv_op3d(x, W, strides=[1,1,1,1,1]):
    return tf.nn.conv3d(x, W, strides=strides, padding='SAME')

def deconv_op3d(x, W, strides=[1,1,1,1,1], padding='SAME'):
    filter_size = list(map(int, W.get_shape()))
    #print(x.get_shape())
    #xs = x.get_shape()
    xs = list(map(int, x.get_shape()))    
    if padding == 'SAME': target_shape = [xs[0], xs[1]*strides[1], xs[2]*strides[2], xs[3]*strides[3], filter_size[-1]]
    elif padding == 'VALID': target_shape = [xs[0], xs[1]*strides[1] + filter_size[0]-1, xs[2]*strides[2] + filter_size[1]-1,
                                             filter_size[-1]]
    return tf.nn.conv3d_transpose(x, W, target_shape, strides, padding)



class GatedCNN():
    def __init__(self, W_shape, fan_in, orientation, gated=True, payload=None, mask=None, simple_maskmode='standard', debug=False,
                 activation=True, conditional=None, conditional_image=None, cfilter_size=1, gatedact='sigmoid', strides=[1,1,1,1,1], deconv=False,
                 convconditional=None):
        self.fan_in = fan_in
        in_dim = self.fan_in.get_shape()[-1]
        if convconditional is not None:
            in_dim += convconditional.get_shape()[-1]
        self.W_shape = [W_shape[0], W_shape[1], W_shape[2], in_dim, W_shape[3]]  
        self.b_shape = W_shape[3]

        self.in_dim = in_dim
        self.payload = payload
        self.mask = mask
        self.activation = activation
        self.conditional = conditional
        self.conditional_image = conditional_image
        self.conv_conditional = convconditional
        self.orientation = orientation
        self.cfilter_size = cfilter_size
        self.debug = debug
        self.strides = strides
        self.deconv = deconv
        self.simple_maskmode = simple_maskmode
        self.gatedact = gatedact
        if gated:
            self.gated_conv()
        else:
            self.simple_conv()


    def gated_conv(self):
        W_f = get_weights3d(self.W_shape, "v_W", self.orientation, mask=self.mask)
        W_g = get_weights3d(self.W_shape, "h_W", self.orientation, mask=self.mask)

        b_f_total = get_bias(self.b_shape, "v_b")
        b_g_total = get_bias(self.b_shape, "h_b")

        if self.conditional_image is not None:
            V_f = tf.layers.conv3d(self.conditional_image, self.in_dim, self.cfilter_size, 
                                   padding='same', use_bias=False, name="ci_f")
            V_g = tf.layers.conv3d(self.conditional_image, self.in_dim, self.cfilter_size, 
                                   padding='same', use_bias=False, name="ci_g")
            b_f_total = b_f_total + V_f
            b_g_total = b_g_total + V_g
            
        if self.conv_conditional is not None:
            conv_f = conv_op3d(tf.concat([self.fan_in, self.conv_conditional], axis=-1), W_f)
            conv_g = conv_op3d(tf.concat([self.fan_in, self.conv_conditional], axis=-1), W_g)
        else:
            conv_f = conv_op3d(self.fan_in, W_f)
            conv_g = conv_op3d(self.fan_in, W_g)
       
        if self.payload is not None:
            conv_f += self.payload
            conv_g += self.payload

        if self.debug: self.fan_out = conv_f + b_f_total
        else:
            if self.gatedact == 'sigmoid':
                self.fan_out = tf.multiply(tf.tanh(conv_f + b_f_total), tf.sigmoid(conv_g + b_g_total))
            elif self.gatedact == 'relu':
                self.fan_out = tf.multiply(tf.tanh(conv_f + b_f_total), tf.nn.relu(conv_g + b_g_total))
            elif self.gatedact == 'leaky_relu':
                self.fan_out = tf.multiply(tf.tanh(conv_f + b_f_total), tf.nn.leaky_relu(conv_g + b_g_total))

        
    def simple_conv(self):
        W = get_weights3d(self.W_shape, "W", self.orientation, mask_mode=self.simple_maskmode, mask=self.mask)
        b = get_bias(self.b_shape, "b")
        if self.deconv: conv = deconv_op3d(self.fan_in, W, strides=self.strides)
        else: conv = conv_op3d(self.fan_in, W, strides=self.strides)
        if self.activation: 
            self.fan_out = tf.nn.leaky_relu(tf.add(conv, b))
        else:
            self.fan_out = tf.add(conv, b)

    def output(self):
        return self.fan_out 


#$#class PixelCNN3D(object):
#$#    def __init__(self, X, full_horizontal=True, h=None):
#$#        self.X = X
#$#        self.X_norm = X
#$#        v_stack_in, h_stack_in, d_stack_in = self.X_norm, self.X_norm, self.X_norm
#$#        self.h = None
#$#        self.im = None #X*0 + 1e-1*np.e
#$#        nlayers = 5
#$#        f_map = 4
#$#        
#$#        for i in range(nlayers):
#$#            filter_size = 3 if i > 0 else 3
#$#            mask = 'b' if i > 0 else 'a'
#$#            residual = True if i > 0 else False
#$#            i = str(i)
#$#            with tf.variable_scope("d_stack"+i):
#$#                d_stack = GatedCNN([filter_size, filter_size, filter_size, f_map], 
#$#                                   d_stack_in, 0, mask=mask, 
#$#                                   conditional=self.h, conditional_image=self.im).output()
#$#                d_stack_in = d_stack
#$#
#$#            with tf.variable_scope("d_stack_1"+i):
#$#                d_stack_1 = GatedCNN([1, 1, 1, f_map], 
#$#                                     d_stack_in, 0, gated=False, mask=None).output()
#$#
#$#            with tf.variable_scope("v_stack"+i):
#$#                v_stack = GatedCNN([filter_size, filter_size, filter_size, f_map], 
#$#                                   v_stack_in, 1, payload = d_stack_1, mask=mask, 
#$#                                   conditional=self.h, conditional_image=self.im).output()
#$#                v_stack_in = v_stack
#$#
#$#            with tf.variable_scope("v_stack_1"+i):
#$#                v_stack_1 = GatedCNN([1, 1, 1, f_map], 
#$#                                     v_stack_in, 1, gated=False, mask=None).output()
#$#                
#$#            with tf.variable_scope("h_stack"+i):
#$#                f0size = filter_size if full_horizontal else 1
#$#                h_stack = GatedCNN([f0size, filter_size, filter_size, f_map], 
#$#                                   h_stack_in, 2, payload=v_stack_1, mask=mask, 
#$#                                   conditional=self.h, conditional_image=self.im).output()
#$#
#$#            with tf.variable_scope("h_stack_1"+i):
#$#                h_stack_1 = GatedCNN([1, 1, 1, f_map],
#$#                                     h_stack, 2, gated=False, mask=None).output()
#$#                if residual:
#$#                    h_stack_1 += h_stack_in # Residual connection
#$#                h_stack_in = h_stack_1
#$#
#$#        
#$#        with tf.variable_scope("fc_1"):
#$#            fc1 = GatedCNN([1, 1, 1, f_map], h_stack_in, orientation=None, gated=False, mask='b').output()
#$#
#$#        with tf.variable_scope("fc_2"):
#$#            self.fc2 = GatedCNN([1, 1, 1, 1], fc1, orientation=None, gated=False, mask='b', activation=False).output()
#$#        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2, labels=self.X))
#$#        self.pred = tf.nn.sigmoid(self.fc2)
#$#


def PixelCNN3Dlayer(i, X, f_map=8, full_horizontal=True, h=None, filter_size=None, 
                    conditional=None, conditional_im=None, cfilter_size=None, debug=False, gatedact='sigmoid', gated=True, convconditional=None):

    if len(X) == 1:
        X_norm = X[0]
        d_stack_in, v_stack_in, h_stack_in = X_norm, X_norm, X_norm
    else:
        d_stack_in, v_stack_in, h_stack_in = X

    if filter_size is None:
        filter_size = 3 if i > 0 else 5
    if cfilter_size is None:
        cfilter_size = filter_size
    mask = 'b' if i > 0 else 'a'
    residual = True if i > 0 else False
    i = str(i)
    with tf.variable_scope("d_stack"+i):
        d_stack = GatedCNN([filter_size, filter_size, filter_size, f_map], 
                           d_stack_in, 0, mask=mask, conditional=conditional, conditional_image=conditional_im,
                           convconditional=convconditional, cfilter_size=cfilter_size, debug=debug, gatedact=gatedact, gated=gated).output()
        
        d_stack_in = d_stack

    with tf.variable_scope("d_stack_1"+i):
        d_stack_1 = GatedCNN([1, 1, 1, f_map], 
                             d_stack_in, 0, gated=False, mask=None).output()

    with tf.variable_scope("v_stack"+i):
        v_stack = GatedCNN([filter_size, filter_size, filter_size, f_map], 
                           v_stack_in, 1, payload = d_stack_1, mask=mask, conditional=conditional, conditional_image=conditional_im,
                           convconditional=convconditional, cfilter_size=cfilter_size, debug=debug, gatedact=gatedact, gated=gated).output()
        v_stack_in = v_stack

    with tf.variable_scope("v_stack_1"+i):
        v_stack_1 = GatedCNN([1, 1, 1, f_map], 
                             v_stack_in, 1, gated=False, mask=None).output()

    with tf.variable_scope("h_stack"+i):
        f0size = filter_size if full_horizontal else 1
        h_stack = GatedCNN([f0size, filter_size, filter_size, f_map], 
                           h_stack_in, 2, payload=v_stack_1, mask=mask, conditional=conditional, conditional_image=conditional_im,
                           convconditional=convconditional, cfilter_size=cfilter_size, debug=debug, gatedact=gatedact, gated=gated).output()

    with tf.variable_scope("h_stack_1"+i):
        h_stack_1 = GatedCNN([1, 1, 1, f_map],
                             h_stack, 2, gated=False, mask=None).output()
        if residual:
            h_stack_1 += h_stack_in # Residual connection
        h_stack_in = h_stack_1


    return d_stack_in, v_stack_in, h_stack_in





def PixelCNN3Dlayer_dn(i, X,  _horizontal=True, h=None, filter_size=None):

    if len(X) == 1:
        X_norm = X[0]
        d_stack_in, v_stack_in, h_stack_in = X_norm, X_norm, X_norm
    else:
        d_stack_in, v_stack_in, h_stack_in = X

    d_stack = d_stack_in[:, ::2, ::2, ::2, :]
    h_stack = h_stack_in[:, ::2, ::2, ::2, :]
    v_stack = v_stack_in[:, ::2, ::2, ::2, :]

##    if filter_size is None:
##        filter_size = 3 if i > 0 else 5
##    mask = 'b' if i > 0 else 'a'
##    f_map = int(d_stack_in.get_shape()[-1])
##    #residual = True if i > 0 else False
##    i = str(i)
##
##    with tf.variable_scope("d_stack_dn"+i):
##        d_stack = GatedCNN([filter_size, filter_size, filter_size, f_map], 
##                           d_stack_in, 0, gated=False, mask=mask, strides=[1, 2, 2, 2, 1]).output()
##
##    with tf.variable_scope("v_stack_dn"+i):
##        v_stack = GatedCNN([filter_size, filter_size, filter_size, f_map], 
##                           v_stack_in, 1, gated=False, mask=mask, strides=[1, 2, 2, 2, 1]).output()
##
##    with tf.variable_scope("h_stack_dn"+i):
##        h_stack = GatedCNN([filter_size, filter_size, filter_size, f_map], 
##                           h_stack_in, 2, gated=False, mask=mask, strides=[1, 2, 2, 2, 1]).output()
##
    return d_stack, v_stack, h_stack
##


def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        #copy = tf.identity(out)
        for i in range(dim, 0, -1):
            #out = tf.concat([out, tf.zeros_like(out)], i)
            out = tf.concat([out, tf.identity(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def PixelCNN3Dlayer_up(i, X,  _horizontal=True, h=None, filter_size=None):


    if len(X) == 1:
        X_norm = X[0]
        d_stack_in, v_stack_in, h_stack_in = X_norm, X_norm, X_norm
    else:
        d_stack_in, v_stack_in, h_stack_in = X

    d_stack = unpool(d_stack_in)
    h_stack = unpool(h_stack_in)
    v_stack = unpool(v_stack_in)

##    if filter_size is None:
##        filter_size = 3 if i > 0 else 5
##    mask = 'b' if i > 0 else 'a'
##    f_map = int(d_stack_in.get_shape()[-1])
##    #residual = True if i > 0 else False
##    i = str(i)
##
##    with tf.variable_scope("d_stack_dn"+i):
##        d_stack = GatedCNN([filter_size, filter_size, filter_size, f_map], 
##                           d_stack_in, 0, gated=False, mask=mask, simple_maskmode='standard', strides=[1, 2, 2, 2, 1], deconv=True).output()
##
##    with tf.variable_scope("v_stack_dn"+i):
##        v_stack = GatedCNN([filter_size, filter_size, filter_size, f_map], 
##                           v_stack_in, 1, gated=False, mask=mask, simple_maskmode='standard', strides=[1, 2, 2, 2, 1], deconv=True).output()
##
##    with tf.variable_scope("h_stack_dn"+i):
##        h_stack = GatedCNN([filter_size, filter_size, filter_size, f_map], 
##                           h_stack_in, 2, gated=False, mask=mask, simple_maskmode='standard', strides=[1, 2, 2, 2, 1], deconv=True).output()
##
    return d_stack, v_stack, h_stack

##
