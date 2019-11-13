import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#
import sys, os
sys.path.append('./utils/')
import tools
import datatools as dtools
from time import time
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#
import tensorflow as tf
import tensorflow_hub as hub





#############################
seed_in = 3
from numpy.random import seed
seed(seed_in)
from tensorflow import set_random_seed
set_random_seed(seed_in)



bs = 400
nc, ncf = 128, 512
step, stepf = 10, 40
path = '../data/z00/'
ftype = 'L%04d_N%04d_S%04d_%02dstep/'
ftypefpm = 'L%04d_N%04d_S%04d_%02dstep_fpm/'
numd = 1e-3
num = int(numd*bs**3)
R1 = 3
R2 = 3*1.2
kny = np.pi*nc/bs
kk = tools.fftk((nc, nc, nc), bs)

#############################

pad = int(0)
masktype = 'constant'
dependence = None
suff = 'pad%d-cic-allnn-cmask-pois4normmix-monp'%pad
savepath = '../models/n10/%s/module/'%suff

ftname = ['cic']
tgname = ['pnn', 'mnnnomean']
nchannels = len(ftname)
ntargets = len(tgname)


def get_meshes(seed, galaxies=False):
    mesh = {}
    mesh['s'] = tools.readbigfile(path + ftypefpm%(bs, nc, seed, step) + 'mesh/s/')
    mesh['cic'] = np.load(path + ftypefpm%(bs, nc, seed, step) + 'mesh/d.npy')
#    partp = tools.readbigfile(path + ftypefpm%(bs, nc, seed, step) + 'dynamic/1/Position/')
#    mesh['cic'] = tools.paintcic(partp, bs, nc)
#    mesh['logcic'] = np.log(1 + mesh['cic'])
#    mesh['decic'] = tools.decic(mesh['cic'], kk, kny)
#    mesh['R1'] = tools.fingauss(mesh['cic'], kk, R1, kny)
#    mesh['R2'] = tools.fingauss(mesh['cic'], kk, R2, kny)
#    mesh['GD'] = mesh['R1'] - mesh['R2']
#
    hmesh = {}
    hposall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]    
    massall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/Mass/')[1:].reshape(-1)*1e10
    hposd = hposall[:num].copy()
    massd = massall[:num].copy()
    hmesh['pnn'] = tools.paintnn(hposd, bs, nc)
    hmesh['mnn'] = tools.paintnn(hposd, bs, nc, massd)
    hmesh['mnnnomean'] =  (hmesh['mnn'])/hmesh['mnn'].mean()
    #hmesh['pcic'] = tools.paintcic(hposd, bs, nc)
    #hmesh['mcic'] = tools.paintcic(hposd, bs, nc, massd)
    #hmesh['mcicnomean'] =  (hmesh['mcic'])/hmesh['mcic'].mean()
    #hmesh['mcicovd'] =  (hmesh['mcic'] - hmesh['mcic'].mean())/hmesh['mcic'].mean()
    #hmesh['mcicovdR3'] = tools.fingauss(hmesh['mcicovd'], kk, R1, kny)    
    #hmesh['pcicovd'] =  (hmesh['pcic'] - hmesh['pcic'].mean())/hmesh['pcic'].mean()
    #hmesh['pcicovdR3'] = tools.fingauss(hmesh['pcicovd'], kk, R1, kny)    
    #hmesh['lmnn'] = np.log(logoffset + hmesh['mnn'])

    return mesh, hmesh
    

    

#####

#
tf.reset_default_graph()

files = os.listdir(savepath)
paths = [os.path.join(savepath, basename) for basename in files]
modpath =  max(paths, key=os.path.getctime)
print(modpath)

module = hub.Module(modpath+'/likelihood/')
xx = tf.placeholder(tf.float32, shape=[None, None, None, None, len(ftname)], name='input')
yy = tf.placeholder(tf.float32, shape=[None, None, None, None, len(tgname)], name='labels')
locpos = module(dict(features=xx, labels=yy), as_dict=True)['locpos']
logitspos = module(dict(features=xx, labels=yy), as_dict=True)['logitspos']
scalepos = module(dict(features=xx, labels=yy), as_dict=True)['scalepos']
rawsamplepos = module(dict(features=xx, labels=yy), as_dict=True)['rawsamplepos']
rawsamples = module(dict(features=xx, labels=yy), as_dict=True)['rawsample']
samples = module(dict(features=xx, labels=yy), as_dict=True)['sample']
loglik = module(dict(features=xx, labels=yy), as_dict=True)['loglikelihood']
pred_mask = module(dict(features=xx, labels=yy), as_dict=True)['pred_mask']






vmeshes = {}
shape = [nc,nc,nc]
kk = tools.fftk(shape, bs)
kmesh = sum(i**2 for i in kk)**0.5

with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())

    for seed in [100]:
        pass
    seed = 100    
    batch = 1
    
    vmeshes[seed] = get_meshes(seed)
    xxm = np.stack([np.pad(vmeshes[seed][0][i], pad, 'wrap') for i in ftname], axis=-1)
    #yym = np.stack([np.pad(vmeshes[seed][1]['pnncen'], pad, 'wrap'), np.pad(vmeshes[seed][1]['pnnsat'], pad, 'wrap')], axis=-1)
    yym = np.stack([vmeshes[seed][1][i] for i in tgname], axis=-1)
    print('xxm, yym shape = ', xxm.shape, yym.shape)

    #logits = np.squeeze(sess.run(logitspos, feed_dict={xx:np.expand_dims(xxm, 0), yy:np.expand_dims(yym, 0)}))
    #loc = np.squeeze(sess.run(locpos, feed_dict={xx:np.expand_dims(xxm, 0), yy:np.expand_dims(yym, 0)}))
    #scale = np.squeeze(sess.run(scalepos, feed_dict={xx:np.expand_dims(xxm, 0), yy:np.expand_dims(yym, 0)}))

    predmask = np.squeeze(sess.run(pred_mask, feed_dict={xx:np.expand_dims(xxm, 0), yy:np.expand_dims(yym, 0)}))
    rawpredspos = np.squeeze(sess.run(rawsamplepos, feed_dict={xx:np.expand_dims(xxm, 0), yy:np.expand_dims(yym, 0)}))
    rawpreds = np.squeeze(sess.run(rawsamples, feed_dict={xx:np.expand_dims(xxm, 0), yy:np.expand_dims(yym, 0)}))
    preds = np.squeeze(sess.run(samples, feed_dict={xx:np.expand_dims(xxm, 0), yy:np.expand_dims(yym, 0)}))

#    print(rawpredspos.shape)
#    print(rawpredspos)
#    print(rawpredspos.min(), rawpredspos.max())
#    print(predmask.min(), predmask.max())

#    print(rawpreds.shape)
#    print(rawpreds[0].min(), rawpreds[0].max())
#
#    print(preds.shape)
#    print(preds[0].min(), preds[0].max())
#
    vmeshes[seed][0]['predict'] = preds
    vmeshes[seed][0]['rawpredict'] = rawpreds

    print('Truth : ', np.unique(vmeshes[seed][1]['pnn'], return_counts=True))
    print('RawSamplePos : ', np.unique(rawpredspos, return_counts=True)) 
    print('Sample : ', np.unique(vmeshes[seed][0]['predict'][0], return_counts=True)) 
    print('RawSample : ', np.unique(vmeshes[seed][0]['rawpredict'][0], return_counts=True)) #Not sure why this is not the same as rawsamplepos
    


    ##############################

    ##Power spectrum
    yy = ['pos', 'mass']
    for iy in range(2):
        fig, axar = plt.subplots(2, 2, figsize = (8, 8))
        ax = axar[0]
        predict, hpmeshd = vmeshes[seed][0]['predict'][...,iy] , np.stack([vmeshes[seed][1][i] for i in tgname], axis=-1)[...,iy]
        print(predict.shape, hpmeshd.shape)
        k, pkpred = tools.power(predict/predict.mean(), boxsize=bs, k=kmesh)
        k, pkhd = tools.power(hpmeshd/hpmeshd.mean(), boxsize=bs, k=kmesh)
        k, pkhx = tools.power(hpmeshd/hpmeshd.mean(), predict/predict.mean(), boxsize=bs, k=kmesh)    
        ##
        ax[0].semilogx(k[1:], pkpred[1:]/pkhd[1:], label=seed)
        ax[1].semilogx(k[1:], pkhx[1:]/(pkpred[1:]*pkhd[1:])**0.5)

        for axis in ax.flatten():
            axis.legend(fontsize=14)
            axis.set_yticks(np.arange(0, 1.2, 0.1))
            axis.grid(which='both')
            axis.set_ylim(0.,1.1)
        ax[0].set_ylabel('Transfer function', fontsize=14)
        ax[1].set_ylabel('Cross correlation', fontsize=14)
        #
        #
        ax = axar[1]
        vmin, vmax = 0, (hpmeshd[:, :, :].sum(axis=0)).max()
        im = ax[0].imshow(predict[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        im = ax[1].imshow(hpmeshd[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        ax[0].set_title('Prediction', fontsize=15)
        ax[1].set_title('Truth', fontsize=15)
        plt.savefig('./vpredict-%s.png'%( yy[iy]))
        plt.show()

        plt.figure()
        plt.hist(hpmeshd.flatten(), range=(-1, 20), bins=100, label='target', alpha=0.8)
        plt.hist(predict.flatten(),  range=(-1, 20), bins=100, label='prediict', alpha=0.5)
        plt.legend()
        plt.yscale('log')
        plt.savefig('./hist-%s.png'%( yy[iy]))
        plt.show()
            ##
