import numpy as np
import sys, os
from scipy.optimize import minimize
import json
import matplotlib.pyplot as plt

#
sys.path.append('./utils')
import tools


#
bs, ncf, stepf = 400, 512, 40
path = '../data/z00/'
ftype = 'L%04d_N%04d_S%04d_%02dstep/'
ftypefpm = 'L%04d_N%04d_S%04d_%02dstep_fpm/'
mm = np.load('../data/Illustris_halo_groupmass.npy').T

mh = mm[1]*1e10
ms = mm[2]*1e10


def getstellar(mbins):
    scount, smass, lsstd = np.zeros_like(mbins), np.zeros_like(mbins), np.zeros_like(mbins)
    hmass = np.zeros_like(mbins)

    for i in range(mbins.size-1):
        if i == mbins.size-1: mask = (mm[1]*1e10 > mbins[i])
        else: mask = (mm[1]*1e10 > mbins[i]) & (mm[1]*1e10<mbins[i+1])
        scount[i] = mask.sum() 
        smass[i] = mm[2][mask].mean()*1e10
        #sstd[i] = mm[2][mask].std()*1e10
        lsstd[i] = np.log(mm[2][mask]*1e10).std()
        hmass[i] = mm[1][mask].mean()*1e10
    
    return scount, smass, lsstd, hmass

def fitstellar(p, smass, hmass, rety=False):
    p0, p1 = p
    yy = p1*np.log(hmass)+p0
    if rety: return np.exp(yy)
    return sum((np.log(smass[:-1]) - yy[:-1])**2)

def fitscatter(p, hmass, rstd, rety=False):
    p0, p1, p2 = p
    xx = np.log(hmass)
    yy = p0 + p1*xx + p2*xx**2
    if rety: return yy
    return sum((yy[:-1] - rstd[:-1])**2)


def dofit():
    mbins = 10**np.arange(12, 14, 0.1)
    scount, smass, lsstd, hmass = getstellar(mbins)
    pp = minimize(lambda p: fitstellar(p, smass, hmass), [1, 1])
    #pps = minimize(lambda p: fitscatter(p, hmass, sstd/smass), [0.3, 0.0, .0])
    pps = minimize(lambda p: fitscatter(p, hmass, lsstd), [0.3, 0.0, .0])

    fname = '../data/stellar.json'
    data = {'stellarfit':list(pp.x), 'scatterfit':list(pps.x)} 
    data['mbins'] = list(mbins)
    data['NOTE'] = 'Fit b/w range 1e12, 1e14'
    with open(fname, "w") as write_file:
        json.dump(data, write_file, indent=4)



def scattercatalog(seed, mmin=1e12):
    hmass = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/Mass/')[1:].reshape(-1)*1e10
    print(hmass.max()/1e12, hmass.min()/1e12)
    with open('../data/stellar.json', "r") as read_file:
        p = json.load(read_file)
    mbins = p['mbins']
    pm = p['stellarfit']
    ps = p['scatterfit']
    print(pm, ps)
    
    smassmean = fitstellar(pm, None, hmass, True)
    smasssig = fitscatter(ps, hmass, None, True)
    print(fitstellar(pm,  None, 1e12,True))
    print(fitscatter(ps,  1e12, None, True))

    smasssig[smasssig < 0.1] = 0.1
    np.random.seed(seed)
    scatter = np.random.normal(scale=smasssig)
    smass = np.exp(np.log(smassmean) + scatter)
    mask = hmass >= mmin
    smass[~mask] = -999
    np.save(path + ftype%(bs, ncf, seed, stepf) + '/stellarmass', smass)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    axis = ax[0]
    axis.plot(hmass[mask], smass[mask], '.')
    axis.plot(hmass[mask], smassmean[mask], '.')
    axis.loglog()
    axis.grid()
    axis.set_title('FastPM')

    axis = ax[1]
    axis.plot(mh[mh>mmin], ms[mh>mmin], '.')
    axis.plot(hmass[mask], smassmean[mask], '.')
    axis.loglog()
    axis.grid()
    axis.set_title('Illustris')
    plt.savefig(path + ftype%(bs, ncf, seed, stepf) + '/stellarmass.png')
    plt.close()
    
if __name__=='__main__':

    if os.path.isfile('../data/stellar.json'): print('Stellar fit exits')
    else: dofit()
    dofit()
    for seed in range(100, 1000, 100):
        scattercatalog(seed)

