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
from tensorflow.contrib.slim import add_arg_scope
from layers import wide_resnet
import tensorflow_hub as hub


import models
import logging
from datetime import datetime

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
seeds = [100, 200, 300, 400, 500, 600, 700]
vseeds = [100, 300, 800, 900]
#seeds = [100]
#vseeds = [100]

#############################

logoffset = 1e-3
distribution = 'logistic'
n_mixture = 4
pad = int(0)
masktype = 'constant'
suff = 'pad%d-cic-cmask-pnn-4mix'%pad
#masktype = 'vary'
#suff = 'pad%d-vmask-mscale-4mix'%pad

#fname = open('../models/n10/README', 'a+', 1)
#fname.write('%s \t :\n\tModel to predict halo position likelihood in halo_logistic with data supplemented by size=8, 16, 32, 64, 128; rotation with probability=0.5 and padding the mesh with 2 cells. Also reduce learning rate in piecewise constant manner. n_y=1 and high of quntized distribution to 3. Init field as 1 feature & high learning rate\n'%suff)
#fname.close()

savepath = '../models/n10/%s/'%suff
try : os.makedirs(savepath)
except: pass


fname = open(savepath + 'log', 'w+', 1)
#fname = None
num_cubes= 500
cube_sizes = np.array([16, 32, 64, 128]).astype(int)
#cube_sizes = np.array([32]).astype(int)
nsizes = len(cube_sizes)
cube_sizesft = (cube_sizes + 2*pad).astype(int)
max_offset = nc - cube_sizes
ftname = ['cic']
tgname = ['pnn']
nchannels = len(ftname)
ntargets = len(tgname)
modelname = '_mdn_mask_model_fn'
modelfunc = getattr(models, modelname)

batch_size=64
rprob = 0.5

print('Features are : ', ftname, file=fname)
print('Target are : ', tgname, file=fname)
print('Model Name : ', modelname, file=fname)
print('Distribution : ', distribution, file=fname)
print('Masktype : ', masktype, file=fname)
print('No. of components : ', n_mixture, file=fname)
print('Pad with : ', pad, file=fname)
print('Rotation probability = %0.2f'%rprob, file=fname)
fname.close()

#############################
##Read data and generate meshes



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



def generate_training_data():
    meshes = {}
    cube_features, cube_target = [[] for i in range(len(cube_sizes))], [[] for i in range(len(cube_sizes))]

    for seed in seeds:

        mesh, hmesh = get_meshes(seed)
        meshes[seed] = [mesh, hmesh]

        print('All the mesh have been generated for seed = %d'%seed)

        #Create training voxels
        ftlist = [mesh[i].copy() for i in ftname]
        ftlistpad = [np.pad(i, pad, 'wrap') for i in ftlist]
        targetmesh = [hmesh[i].copy() for i in tgname]

        for i, size in enumerate(cube_sizes):
            print('For size = ', size)
            if size==nc:
                features = [np.stack(ftlistpad, axis=-1)]
                target = [np.stack(targetmesh, axis=-1)]
            else:
                numcubes = int(num_cubes/size*4)
                features, target = dtools.randomvoxels(ftlistpad, targetmesh, numcubes, max_offset[i], 
                                                size, cube_sizesft[i], seed=seed, rprob=0)
            cube_features[i] = cube_features[i] + features
            cube_target[i] = cube_target[i] + target

     
    for i in range(cube_sizes.size):
        cube_target[i] = np.stack(cube_target[i],axis=0)
        cube_features[i] = np.stack(cube_features[i],axis=0)
        print(cube_features[i].shape, cube_target[i].shape)

    return meshes, cube_features, cube_target



#############################

class MDNEstimator(tf.estimator.Estimator):
    """An estimator for distribution estimation using Mixture Density Networks.
    """

    def __init__(self,
                 n_y,
                 n_mixture,
                 optimizer=tf.train.AdamOptimizer,
                 dropout=None,
                 model_dir=None,
                 config=None):
        """Initializes a `MDNEstimator` instance.
        """

        def _model_fn(features, labels, mode):
            return modelfunc(features, labels, 
                                 nchannels, n_y, n_mixture, dropout,
                                             optimizer, mode, pad, distribution=distribution, masktype=masktype)

        super(self.__class__, self).__init__(model_fn=_model_fn,
                                             model_dir=model_dir,
                                             config=config)





def mapping_function(inds):
    def extract_batch(inds):
        
        isize = np.random.choice(len(cube_sizes), 1, replace=True)[0]
        batch = int(batch_size*8/cube_sizes[isize])
        if cube_sizes[isize]==nc : batch = 1
        inds = inds[:batch]
        trainingsize = cube_features[isize].shape[0]
        inds[inds >= trainingsize] =  (inds[inds >= trainingsize])%trainingsize
        
        features = cube_features[isize][inds].astype('float32')
        targets = cube_target[isize][inds].astype('float32')
        
        for i in range(batch):
            nrotations=0
            while (np.random.random() < rprob) & (nrotations < 3):
                nrot, ax0, ax1 = np.random.randint(0, 3), *np.random.permutation((0, 1, 2))[:2]
                features[i] = np.rot90(features[i], nrot, (ax0, ax1))
                targets[i] = np.rot90(targets[i], nrot, (ax0, ax1))
                nrotations +=1
# #             print(isize, i, nrotations, targets[i].shape)
# #         print(inds)
        return features, targets
    
    ft, tg = tf.py_func(extract_batch, [inds],
                        [tf.float32, tf.float32])
    return ft, tg

def training_input_fn():
    """Serving input fn for training data"""

    dataset = tf.data.Dataset.range(len(np.array(cube_features)[0]))
    dataset = dataset.repeat().shuffle(1000).batch(batch_size)
    dataset = dataset.map(mapping_function)
    dataset = dataset.prefetch(16)
    return dataset

def testing_input_fn():
    """Serving input fn for testing data"""
    dataset = tf.data.Dataset.range(len(cube_features))
    dataset = dataset.batch(16)
    dataset = dataset.map(mapping_function)
    return dataset


        
#############################################################################
###save


def save_module(model, savepath, max_steps):

    print('\nSave module\n')

    features = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
    labels = tf.placeholder(tf.float32, shape=[None, None, None, None, ntargets], name='labels')
    exporter = hub.LatestModuleExporter("tf_hub", tf.estimator.export.build_raw_serving_input_receiver_fn({'features':features, 'labels':labels},
                                                                       default_batch_size=None))
    modpath = exporter.export(model, savepath + 'module', model.latest_checkpoint())
    modpath = modpath.decode("utf-8") 
    check_module(modpath)
    

#####
def check_module(modpath):
    
    print('\nTest module\n')

    tf.reset_default_graph()
    module = hub.Module(modpath + '/likelihood/')
    xx = tf.placeholder(tf.float32, shape=[None, None, None, None, nchannels], name='input')
    yy = tf.placeholder(tf.float32, shape=[None, None, None, None, ntargets], name='labels')
    samples = module(dict(features=xx, labels=yy), as_dict=True)['sample']
    loglik = module(dict(features=xx, labels=yy), as_dict=True)['loglikelihood']

    preds = {}
    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())

        for seed in vseeds:
            xxm = np.stack([np.pad(vmeshes[seed][0][i], pad, 'wrap') for i in ftname], axis=-1)
            #yym = np.stack([np.pad(vmeshes[seed][1]['pnncen'], pad, 'wrap'), np.pad(vmeshes[seed][1]['pnnsat'], pad, 'wrap')], axis=-1)
            yym = np.stack([vmeshes[seed][1][i] for i in tgname], axis=-1)
            print('xxm, yym shape = ', xxm.shape, yym.shape)
            preds[seed] = sess.run(samples, feed_dict={xx:np.expand_dims(xxm, 0), yy:np.expand_dims(yym, 0)})
            preds[seed] = np.squeeze(preds[seed])
            vmeshes[seed][0]['predict'] = preds[seed][:, :, :]


    ##############################
    ##Power spectrum
    shape = [nc,nc,nc]
    kk = tools.fftk(shape, bs)
    kmesh = sum(i**2 for i in kk)**0.5

    fig, axar = plt.subplots(2, 2, figsize = (8, 8))
    ax = axar[0]
    for seed in vseeds:
        for i, key in enumerate(['']):
            predict, hpmeshd = vmeshes[seed][0]['predict%s'%key] , vmeshes[seed][1][tgname[0]], 
            k, pkpred = tools.power(predict/predict.mean(), boxsize=bs, k=kmesh)
            k, pkhd = tools.power(hpmeshd/hpmeshd.mean(), boxsize=bs, k=kmesh)
            k, pkhx = tools.power(hpmeshd/hpmeshd.mean(), predict/predict.mean(), boxsize=bs, k=kmesh)    
        ##
            ax[0].semilogx(k[1:], pkpred[1:]/pkhd[1:], label=seed)
            ax[1].semilogx(k[1:], pkhx[1:]/(pkpred[1:]*pkhd[1:])**0.5)
            ax[0].set_title(key, fontsize=12)

    for axis in ax.flatten():
        axis.legend(fontsize=14)
        axis.set_yticks(np.arange(0, 1.2, 0.1))
        axis.grid(which='both')
        axis.set_ylim(0.,1.1)
    ax[0].set_ylabel('Transfer function', fontsize=14)
    ax[1].set_ylabel('Cross correlation', fontsize=14)
    #
    ax = axar[1]
    for i, key in enumerate([ '']):
        predict, hpmeshd = vmeshes[seed][0]['predict%s'%key] , vmeshes[seed][1][tgname[0]], 
        vmin, vmax = 1, (hpmeshd[:, :, :].sum(axis=0)).max()
        im = ax[0].imshow(predict[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        im = ax[1].imshow(hpmeshd[:, :, :].sum(axis=0), vmin=vmin, vmax=vmax)
        ax[0].set_title(key, fontsize=15)
    ax[0].set_title('Prediction', fontsize=15)
    ax[1].set_title('Truth', fontsize=15)
    plt.savefig(savepath + '/vpredict%d.png'%max_steps)
    plt.show()

    plt.figure()
    plt.hist(hpmeshd.flatten(), range=(-1, 20), bins=100, label='target', alpha=0.8)
    plt.hist(predict.flatten(),  range=(-1, 20), bins=100, label='prediict', alpha=0.5)
    plt.legend()
    plt.savefig(savepath + '/hist%d.png'%max_steps)
    plt.show()
    ##

#


############################################################################
#############---------MAIN---------################

meshes, cube_features, cube_target = generate_training_data()
vmeshes = {}
for seed in vseeds: vmeshes[seed] = get_meshes(seed)

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
try: os.makedirs(savepath + '/logs/')
except: pass
logfile = datetime.now().strftime('logs/tflogfile_%H_%M_%d_%m_%Y.log')
fh = logging.FileHandler(savepath + logfile)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)


for max_steps in [50, 100, 500, 1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 50000, 60000, 70000]:
#for max_steps in [100]+list(np.arange(5e3, 7.1e4, 5e3, dtype=int)):
    print('For max_steps = ', max_steps)
    tf.reset_default_graph()
    run_config = tf.estimator.RunConfig(save_checkpoints_steps = 2000)

    model =  MDNEstimator(n_y=ntargets, n_mixture=n_mixture, dropout=0.95,
                      model_dir=savepath + 'model', config = run_config)

    model.train(training_input_fn, max_steps=max_steps)
    #save_module(model, savepath, max_steps)
    f = open(savepath + 'model/checkpoint')
    lastpoint = int(f.readline().split('-')[-1][:-2])
    f.close()
    if lastpoint > max_steps:
        print('Don"t save')
        print(lastpoint)
    else:
        print("Have to save")
        save_module(model, savepath, max_steps)
