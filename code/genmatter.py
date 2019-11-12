import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import flowpm
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline

sys.path.append('./utils')
import tools

bs, nc = 400, 128
nsteps = 10
ainit = 0.1

stages = np.linspace(ainit, 1.0, nsteps, endpoint=True)
pk = np.loadtxt('../data/ics_matterpow_0.dat')
ipklin = iuspline(pk[:, 0], pk[:, 1])

print('loaded')
##
##for seed in range(10, 10000, 10):
##
##    print(seed)
##    path = '../data/make_data_code/L%d-N%d-B1-T%d/S%d/'%(bs, nc, nsteps, seed)
##
##    if os.path.isdir(path ):
##        if not os.path.isfile(path + '/fpm-d'):
##            #ick = tools.readbigfile(path + '/linear/LinearDensityK/')
##            #ic = np.fft.irfftn(ick)*nc**3
##            ic = tools.readbigfile(path + '/mesh/s/')
##            initial_conditions = tf.cast(tf.expand_dims(tf.constant(ic), 0), tf.float32) - 1.
##
##            print(initial_conditions)
##
##            # Sample particles
##            state = flowpm.lpt_init(initial_conditions, a0=ainit)   
##
##            # Evolve particles down to z=0
##            final_state = flowpm.nbody(state, stages, nc)         
##
##            # Retrieve final density field
##            final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), final_state[0])
##
##
##            with tf.Session() as sess:
##                #ic, sim = sess.run([initial_conditions, final_field])
##                sim = sess.run(final_field)
##
##            print(ic.mean())
##            #np.save(path + '/fpm-s', np.squeeze(ic))
##            np.save(path + '/fpm-d', np.squeeze(sim))
##        else:
##            print(path + '/fpm-d' + ' exists')
##    else:
##        print(path + ' does not exist')
##        
##



for ss in range(100, 1000, 100):

    path = '../data/z00/L%04d_N%04d_S%04d_%dstep/'%(bs, nc,  ss, nsteps)

    ic = np.expand_dims(tools.readbigfile(path + '/mesh/s/').astype(np.float32), axis=0)
    print(ic.shape)

    initial_conditions = tf.cast(tf.constant(ic), tf.float32) 
        
    print(initial_conditions)

    # Sample particles
    state = flowpm.lpt_init(initial_conditions, a0=ainit)   

    # Evolve particles down to z=0
    final_state = flowpm.nbody(state, stages, nc)         

    # Retrieve final density field
    final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), final_state[0])


    with tf.Session() as sess:
        #ic, sim = sess.run([initial_conditions, final_field])
        sim = sess.run(final_field)

    print(sim.shape)
    np.save(path + '/mesh/d', np.squeeze(sim))

