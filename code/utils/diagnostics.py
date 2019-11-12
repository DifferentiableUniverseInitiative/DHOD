import numpy as np
import matplotlib.pyplot as plt
import numpy
import os, sys

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import tensorflow_hub as hub

sys.path.append('../flowpm/')
from background import *
import tfpm 
import tfpmfuncs as tfpf
from tfpmconfig import Config

sys.path.append('../utils/')
import tools
import datatools as dtools
#Generate DATA

dpath = './../../data/z00/'
dpath = '//project/projectdirs/astro250/chmodi/cosmo4d/data/z00/'
ftype = 'L%04d_N%04d_S%04d_%02dstep/'


def graphpm(config, verbose=True, initlin=False):
    '''return graph to do pm simulation
    if initlin is False, the returned graph generates initial conditions
    if initlin is True, the returned graph has a placeholder'''
    bs, nc = config['boxsize'], config['nc']
    g = tf.Graph()
    with g.as_default():

        linmesh = tf.placeholder(tf.float32, (nc, nc, nc), name='linmesh')
        if initlin:
            linear = tf.Variable(0.)
            linear = tf.assign(linear, linmesh, validate_shape=False, name='linear')
        else:
            linear = tfpm.linfield(config, name='linear')
        icstate = tfpm.lptinit(linear, config, name='icstate')
        fnstate = tfpm.nbody(icstate, config, verbose=verbose, name='fnstate')
        final = tf.zeros_like(linear)
        final = tfpf.cic_paint(final, fnstate[0], boxsize=config['boxsize'], name='final')
        tf.add_to_collection('pm', [linear, icstate, fnstate, final])
    return g



def genpm(config, linmesh=None, ofolder=None, verbose=True):
    '''do pm sim to generate final matter field'''
    ##Generate Data
    bs, nc = config['boxsize'], config['nc']
    if linmesh is None:
        g = graphpm(config, initlin=False)
        linmesh = np.zeros((nc, nc, nc))
    else:
        g = graphpm(config, initlin=True, verbose=verbose)

    with tf.Session(graph=g) as session:
        session.run(tf.global_variables_initializer())
        linmesh_t = g.get_tensor_by_name('linmesh:0')
        linear_t = g.get_tensor_by_name('linear:0')
        final_t = g.get_tensor_by_name('final:0')
        linear, final = session.run([linear_t, final_t], {linmesh_t:linmesh})

    return linear, final




def graphlintomod(config, modpath, pad=False, ny=1):
    '''return graph to do pm sim and then sample halo positions from it'''
    bs, nc = config['boxsize'], config['nc']

    g = tf.Graph()
    with g.as_default():
        module = hub.Module(modpath)

        linmesh = tf.placeholder(tf.float32, (nc, nc, nc), name='linmesh')
        datamesh = tf.placeholder(tf.float32, (nc, nc, nc, ny), name='datamesh')
        
        #PM
        linear = tf.Variable(0.)
        linear = tf.assign(linear, linmesh, validate_shape=False, name='linear')
        icstate = tfpm.lptinit(linear, config, name='icstate')
        fnstate = tfpm.nbody(icstate, config, verbose=False, name='fnstate')
        final = tf.zeros_like(linear)
        final = tfpf.cic_paint(final, fnstate[0], boxsize=bs, name='final')
        #Sample
        if pad:
            xx = tf.concat((final[-pad:, :, :], final, final[:pad, :, :]), axis=0)
            xx = tf.concat((xx[:, -pad:, :], xx, xx[:, :pad, :]), axis=1)
            xx = tf.concat((xx[:, :, -pad:], xx, xx[:, :, :pad]), axis=2)
            xx = tf.expand_dims(tf.expand_dims(xx, 0), -1)
        else: xx = tf.assign(final)

        yy = tf.expand_dims(datamesh, 0)
        samples = module(dict(features=xx, labels=yy), as_dict=True)['sample']
        samples = tf.identity(samples, name='samples')
        loglik = module(dict(features=xx, labels=yy), as_dict=True)['loglikelihood']
        loglik = tf.identity(loglik, name='loglik')
                
        tf.add_to_collection('inits', [linmesh, datamesh])
        tf.add_to_collection('reconpm', [linear, final, fnstate, samples, loglik])

    return g



def genlintomod(config, modpath, linmesh, datamesh, pad=False):
    '''do pm sim to generate final matter field'''
    ##Generate Data
    bs, nc = config['boxsize'], config['nc']
    ny = datamesh.shape[-1]
    g = graphlintomod(config, modpath, pad=pad, ny=ny)
    print('\nGraph constructed\n')
    
    with tf.Session(graph=g) as session:
        session.run(tf.global_variables_initializer())
        linmesh_t = g.get_tensor_by_name('linmesh:0')
        datamesh_t = g.get_tensor_by_name('datamesh:0')
        linear_t = g.get_tensor_by_name('linear:0')
        final_t = g.get_tensor_by_name('final:0')
        samples_t = g.get_tensor_by_name('samples:0')
        loglik_t = g.get_tensor_by_name('loglik:0')
        linear, final, data, loglik = session.run([linear_t, final_t, samples_t,loglik_t],
                                             {linmesh_t:linmesh, datamesh_t:datamesh})
                         
    return linear, final, data




def savehalofig(truemesh, reconmesh, fname, hgraph, boxsize, title=''):
    '''Given a graph, list of 3 fields in truemesh & recon-init
    create the diagnostic figure,  3X3
    '''
    truelin, truefin, truedata = truemesh

    with tf.Session(graph=hgraph) as sessionh:
        sessionh.run(tf.global_variables_initializer())
        gh = sessionh.graph
        linmesh_t = gh.get_tensor_by_name('linmesh:0')
        datamesh_t = gh.get_tensor_by_name('datamesh:0')
        linear_t = gh.get_tensor_by_name('linear:0')
        final_t = gh.get_tensor_by_name('final:0')
        samples_t = gh.get_tensor_by_name('samples:0')

        linear, final, data = sessionh.run([linear_t, final_t, samples_t],
                                             {linmesh_t:reconmesh, datamesh_t:np.expand_dims(truedata, -1)*0})

    reconmeshlist = [linear, final, data]
    makefig(truemesh, reconmeshlist, fname, boxsize, title)
    
##    fig, ax = plt.subplots(3, 3, figsize = (12,12))
##    meshes = [[truelin, linear], [truefin, final], [truedata, data]]
##    labels = ['Linear', 'Final', 'Data']
##    for i in range(3):
##        m1, m2 = meshes[i][0], meshes[i][1]
##        if m1.mean() < 1e-6:
##            m1, m2 = m1+1, m2+1
##        k, pt = tools.power(m1, boxsize=boxsize)
##        k, pr = tools.power(m2, boxsize=boxsize)
##        k, px = tools.power(m1, m2, boxsize=boxsize)
##        ax[0, 0].semilogx(k, px/(pr*pt)**.5, 'C%d'%i, label=labels[i])
##        ax[0, 1].semilogx(k, pr/pt, 'C%d'%i)
##        ax[0, 2].loglog(k, pt, 'C%d'%i)
##        ax[0, 2].loglog(k, pr, 'C%d--'%i)
##        ax[1, i].imshow(m2.sum(axis=0))
##        ax[2, i].imshow(m1.sum(axis=0))
##    ax[2, 0].set_ylabel('Truth')
##    ax[1, 0].set_ylabel('Recon')
##    ax[0, 0].set_title('Cross Correlation')
##    ax[0, 0].set_ylim(-0.1, 1.1)
##    ax[0, 1].set_title('Transfer Function')
##    ax[0, 1].set_ylim(-0.1, 2)
##    ax[0, 2].set_title('Powers')
##    ax[0, 2].set_ylim(1, 1e5)
##    ax[0, 0].legend()
##    for axis in ax.flatten(): axis.grid(which='both', lw=0.5, color='gray')
##    fig.suptitle(title)
##    fig.tight_layout(rect=[0, 0, 1, 0.95])
##    fig.savefig(fname)
##
def makefig(truemesh, reconmesh, fname, boxsize, title=''):
    '''Given a graph, list of 3 fields in truemesh & recon-init
    create the diagnostic figure,  3X3
    '''
    truelin, truefin, truedata = truemesh
    linear, final, data = reconmesh
    fig, ax = plt.subplots(3, 3, figsize = (12,12))
    meshes = [[truelin, linear], [truefin, final], [truedata, data]]
    labels = ['Linear', 'Final', 'Data']
    for i in range(3):
        m1, m2 = meshes[i][0], meshes[i][1]
        if m1.mean() < 1e-2:
            m1, m2 = m1+1, m2+1
        k, pt = tools.power(m1, boxsize=boxsize)
        k, pr = tools.power(m2, boxsize=boxsize)
        k, px = tools.power(m1, m2, boxsize=boxsize)
        ax[0, 0].semilogx(k, px/(pr*pt)**.5, 'C%d'%i, label=labels[i])
        ax[0, 1].semilogx(k, pr/pt, 'C%d'%i)
        ax[0, 2].loglog(k, pt, 'C%d'%i)
        ax[0, 2].loglog(k, pr, 'C%d--'%i)
        ax[1, i].imshow(m2.sum(axis=0))
        ax[2, i].imshow(m1.sum(axis=0))
    ax[2, 0].set_ylabel('Truth')
    ax[1, 0].set_ylabel('Recon')
    ax[0, 0].set_title('Cross Correlation')
    ax[0, 0].set_ylim(-0.1, 1.1)
    ax[0, 1].set_title('Transfer Function')
    ax[0, 1].set_ylim(-0.1, 2)
    ax[0, 2].set_title('Powers')
    ax[0, 2].set_ylim(1, 1e5)
    ax[0, 0].legend()
    for axis in ax.flatten(): axis.grid(which='both', lw=0.5, color='gray')
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fname)


def savehalofig2(truemesh, reconmesh, fname, hgraph):
    '''Given a graph, list of 3 fields in truemesh & recon-init
    create the diagnostic figure, 4X3
    '''
    truelin, truefin, truedata = truemesh
    with tf.Session(graph=hgraph) as sessionh:
        sessionh.run(tf.global_variables_initializer())
        gh = sessionh.graph
        linmesh_t = gh.get_tensor_by_name('linmesh:0')
        datamesh_t = gh.get_tensor_by_name('datamesh:0')
        linear_t = gh.get_tensor_by_name('linear:0')
        final_t = gh.get_tensor_by_name('final:0')
        samples_t = gh.get_tensor_by_name('samples:0')

        linear, final, data = sessionh.run([linear_t, final_t, samples_t],
                                             {linmesh_t:reconmesh, datamesh_t:np.expand_dims(truedata, -1)*0})

    fig, ax = plt.subplots(4, 3, figsize = (12, 16))
    meshes = [[truelin, linear], [truefin, final], [truedata, data]]
    labels = ['Linear', 'Final', 'Data']
    for i in range(3):
        m1, m2 = meshes[i][0], meshes[i][1]
        if m1.mean() < 1e-6:
            m1, m2 = m1+1, m2+1
        k, pt = tools.power(m1, boxsize=bs)
        k, pr = tools.power(m2, boxsize=bs)
        k, px = tools.power(m1, m2, boxsize=bs)
        ax[0, 0].semilogx(k, px/(pr*pt)**.5, 'C%d'%i, label=labels[i])
        ax[0, 1].semilogx(k, pr/pt, 'C%d'%i)
        ax[0, 2].loglog(k, pt, 'C%d'%i)
        ax[0, 2].loglog(k, pr, 'C%d--'%i)
        im = ax[1, i].imshow(m1.sum(axis=0))
        plt.colorbar(im, ax=ax[1, i])
        im = ax[2, i].imshow(m2.sum(axis=0))
        plt.colorbar(im, ax=ax[2, i])
        im = ax[3, i].imshow((m1-m2).sum(axis=0))
        plt.colorbar(im, ax=ax[3, i])
    ax[1, 0].set_ylabel('Truth')
    ax[2, 0].set_ylabel('Recon')
    ax[0, 0].set_title('Cross Correlation')
    ax[0, 0].set_ylim(-0.1, 1.1)
    ax[0, 1].set_title('Transfer Function')
    ax[0, 1].set_ylim(-0.1, 2)
    ax[0, 2].set_title('Powers')
    ax[0, 2].set_ylim(1, 1e5)
    ax[0, 0].legend()
    for axis in ax.flatten(): axis.grid(which='both', lw=0.5, color='gray')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fname)






#
#
#def savegalfig(truemesh, reconmesh, fname, hgraph):
#
#    truelin, truefin, truedata = truemesh
#    with tf.Session(graph=hgraph) as session:
#        session.run(tf.global_variables_initializer())
#        linmesh_t = g.get_tensor_by_name('linmesh:0')
#        datamesh_t = g.get_tensor_by_name('datamesh:0')
#        linear_t = g.get_tensor_by_name('linear:0')
#        final_t = g.get_tensor_by_name('final:0')
#        samples_t = g.get_tensor_by_name('samples:0')
#
#        linear, final, data = session.run([linear_t, final_t, samples_t],
#                                             {linmesh_t:reconmesh, datamesh_t:reconmesh*0})
#
#    fig, ax = plt.subplots(3, 3, figsize = (12, 8))
#    meshes = [[truelin, linear], [truefin, final], [truedata, data]]
#    for i in range(3):
#        m1, m2 = meshes[i][0], meshes[i][1]
#        k, pt = tools.power(1+m1, boxsize=bs)
#        k, pr = tools.power(1+m2, boxsize=bs)
#        k, px = tools.power(1+m1, 1+m2, boxsize=bs)
#        ax[0, 0].semilogx(k, px/(pr*pt)**.5, 'C%d'%i)
#        ax[0, 1].semilogx(k, pr/pt, 'C%d'%i)
#        ax[0, 2].loglog(k, pt, 'C%d'%i)
#        ax[0, 2].loglog(k, pr, 'C%d--'%i)
#        ax[i, 1].imshow(m1.sum(axis=0))
#        ax[i, 2].imshow(m2.sum(axis=0))
#    ax[1, 0].set_ylabel('Truth')
#    ax[2, 0].set_ylabel('Recon')
#    for axis in ax.flatten(): axis.grid(which='both', lw=0.5, color='gray')
#    fig.tight_layout()
#    fig.savefig(fname)
#




#######################################################
if __name__=="__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    bs, nc = 400, 128
    seed = 100
    step = 5

    config = Config(bs=bs, nc=nc, seed=seed)
#    modpath = '/home/chmodi/Projects/galmodel/code/models/n10/pad2-logistic/module/1546529135/likelihood'
#
#    testlin = np.random.uniform(size=nc**3).reshape(nc, nc, nc)
##
#
#    truelin = tools.readbigfile(dpath + ftype%(bs, nc, seed, step) + 'mesh/s/').astype(np.float32)
#    truefin = tools.readbigfile(dpath + ftype%(bs, nc, seed, step) + 'mesh/d/').astype(np.float32)
#    truedata = dtools.gethalomesh(bs, nc, seed).astype(np.float32)
#    
#    g = graphlintomod(config, modpath, pad=2, ny=1)
#    savehalofig2([truelin, truefin, truedata], truelin, fname='./figs/genlintohpos2.png', hgraph=g, boxsize=bs)
#    
#    linear, final, data = genlintomod(config, modpath, truelin, np.expand_dims(truedata, -1)*0, pad=2)
#    fig, axar = plt.subplots(2, 3, figsize = (12, 8))
#    ax = axar[0]
#    im = ax[0].imshow(truelin.sum(axis=0))
#    plt.colorbar(im, ax=ax[0])
#    im = ax[1].imshow(truefin.sum(axis=0))
#    plt.colorbar(im, ax=ax[1])
#    im = ax[2].imshow(truedata.sum(axis=0))
#    plt.colorbar(im, ax=ax[2])
#    ax[0].set_ylabel('Truth')
#    ax = axar[1]
#    im = ax[0].imshow(linear.sum(axis=0))
#    plt.colorbar(im, ax=ax[0])
#    im = ax[1].imshow(final.sum(axis=0))
#    plt.colorbar(im, ax=ax[1])
#    im = ax[2].imshow(data.sum(axis=0))
#    plt.colorbar(im, ax=ax[2])
#    ax[0].set_ylabel('Sample')
#    plt.savefig('./figs/genlintohpos.png')
#
#

#    ###Test pm without input lin filed
#    print('do without input field')
#    linear, final = genpm(config)
#    fig, ax = plt.subplots(1, 2, figsize = (8, 4))
#    im = ax[0].imshow(linear.sum(axis=0))
#    plt.colorbar(im, ax=ax[0])
#    im = ax[1].imshow(final.sum(axis=0))
#    plt.colorbar(im, ax=ax[1])
#    plt.savefig('./figs/genpm.png')
##

    ###Test pm with input lin filed

    truelin = tools.readbigfile(dpath + ftype%(bs, nc, seed, step) + 'mesh/s/').astype(np.float32)
    truefin = tools.readbigfile(dpath + ftype%(bs, nc, seed, step) + 'mesh/d/').astype(np.float32)

    print('\ndo with input field\n')
    linear, final = genpm(config, linmesh=truelin)
    fig, ax = plt.subplots(2, 3, figsize = (12, 8))
    im = ax[0, 0].imshow(truelin.sum(axis=0))
    plt.colorbar(im, ax=ax[0, 0])
    im = ax[1, 0].imshow(truefin.sum(axis=0))
    plt.colorbar(im, ax=ax[1, 0])
    im = ax[0, 1].imshow(linear.sum(axis=0))
    plt.colorbar(im, ax=ax[0, 1])
    im = ax[1, 1].imshow(final.sum(axis=0))
    plt.colorbar(im, ax=ax[1, 1])
    im = ax[0, 2].imshow((truelin-linear).sum(axis=0))
    plt.colorbar(im, ax=ax[0, 2])
    im = ax[1, 2].imshow((truefin-final).sum(axis=0))
    plt.colorbar(im, ax=ax[1, 2])
    plt.savefig('./figs/genpmlinmesh.png')
