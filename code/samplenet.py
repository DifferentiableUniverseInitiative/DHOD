import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#
import sys, os
sys.path.append('./utils/')
import tools
import datatools as dtools
from time import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#
import tensorflow as tf
import tensorflow_hub as hub





#############################
seed_in = 3
from numpy.random import seed
seed(seed_in)
from tensorflow import set_random_seed
set_random_seed(seed_in)

bss, ncc = [100, 200], [32, 64]
batch_size = [100, 20]
nsteps = 5
cube_sizes = np.array(ncc)
nsizes = len(cube_sizes)
bsnclist = list(zip(bss, ncc))


pad = int(0)
masktype = 'constant'
suff = 'pad%d-cic-mcicnomean-cmask-4normmix'%pad
savepath = '../models/n10/%s/module/'%suff
ftname = ['cic']
tgname = ['mcicnomean']


files = os.listdir(savepath)
paths = [os.path.join(savepath, basename) for basename in files]
modpath =  max(paths, key=os.path.getctime)

print(modpath)



def generate_data(seed, bs, nc):

    j = np.where(cube_sizes == nc)[0][0]
    
    path = '../data/make_data_code/L%d-N%d-B1-T5/S%d/'%(bs, nc, seed)
    #path = '../data/L%d-N%d-B1-T5/S%d/'%(bs, nc, seed)

    mesh = {}
#    mesh['s'] = np.load(path + 'fpm-s.npy')
    mesh['cic'] = np.load(path + 'fpm-d.npy')    
#    mesh['logcic'] = np.log(1 + mesh['cic'])
#    mesh['decic'] = tools.decic(mesh['cic'], kk, kny)
#    mesh['R1'] = tools.fingauss(mesh['cic'], kk, R1, kny)
#    mesh['R2'] = tools.fingauss(mesh['cic'], kk, R2, kny)
#    mesh['GD'] = mesh['R1'] - mesh['R2']
#
    ftlist = [mesh[i].copy() for i in ftname]
    ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
    features = [np.stack(ftlistpad, axis=-1)]

    return features
    

    

#####

#
tf.reset_default_graph()
module = hub.Module(modpath+'/likelihood/')
xx = tf.placeholder(tf.float32, shape=[None, None, None, None, len(ftname)], name='input')
yy = tf.placeholder(tf.float32, shape=[None, None, None, None, len(tgname)], name='labels')
samples = module(dict(features=xx, labels=yy), as_dict=True)['sample']
loglik = module(dict(features=xx, labels=yy), as_dict=True)['loglikelihood']

with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())

    for j in range(nsizes):

        bs, nc = bsnclist[j]
        batch = int(batch_size[j])

        for ss in range(10, 10000, batch*10):
            print(ss)
            seeds = np.arange(ss, ss+batch*10, 10)
        
            try:
                xxm = []
                sskipseed = []
                for iseed, seed in enumerate(seeds):
                    path = '../data/make_data_code/L%d-N%d-B1-T%d/S%d/'%(bs, nc, nsteps, seed)
                    try: xxm.append(generate_data(seed, bs, nc))
                    except Exception as e:
                        print('skip seed :', iseed)
                        print(e)
                        sskipseed.append(iseed)

                xxm = np.concatenate(xxm, axis=0)
                zeros = np.zeros((list(xxm.shape[:-1]) + [len(tgname)]))
                print(xxm.shape)

                preds = np.squeeze(sess.run(samples, feed_dict={xx:xxm, yy:zeros}))

                print(preds.shape)

                for iseed, seed in enumerate(seeds):
                    if iseed in sskipseed:
                        print('skip seed :', iseed)
                        continue
                    path = '../data/make_data_code/L%d-N%d-B1-T%d/S%d/'%(bs, nc, nsteps, seed)
                    np.save(path + '/%s-%s'%(suff, tgname[0]), np.squeeze(preds[iseed]))

            except Exception as e: print(e)



