import numpy as np
import os
import tools
package_path = os.path.dirname(os.path.abspath(__file__))

def randomvoxels(ftlist, targetlist, num_cubes, max_offset, cube_size=32, cube_sizeft=32, seed=100, rprob=0.0):
    '''Generate 'num_cubes' training voxels of 'cube_size' for target and 'cube_sizeft' for features
    from the meshes in 'ftlist' for features and targets in 'target'.
    Rotate voxels with probability 'rprob'
    '''
    np.random.seed(seed)
    rand = np.random.rand

    nchannels = len(ftlist)
    if type(targetlist) == list:
        pass
    else:
        targetlist = [targetlist]
    ntarget = len(targetlist)
    print('Length of targets = ', ntarget)

    cube_features = []
    cube_target = []

    nrotated = 0 
    for it in range(num_cubes):
        #print(it)
        # Extract random cubes from the sim
        offset_x = int(round(rand()*max_offset))
        offset_y = int(round(rand()*max_offset))
        offset_z = int(round(rand()*max_offset))
        x1, x2, x2p = offset_x, offset_x+cube_size, offset_x+cube_sizeft
        y1, y2, y2p = offset_y, offset_y+cube_size, offset_y+cube_sizeft
        z1, z2, z2p = offset_z, offset_z+cube_size, offset_z+cube_sizeft

        features = []
        for i in range(nchannels): features.append(ftlist[i][x1:x2p, y1:y2p, z1:z2p])
        cube_features.append(np.stack(features, axis=-1))
        #
        targets = []
        for i in range(ntarget): targets.append(targetlist[i][x1:x2, y1:y2, z1:z2])
        cube_target.append(np.stack(targets, axis=-1))
        
        rotate = False
        rotation = []
        while (np.random.random() < rprob) & (len(rotation) <= 3):
            rotate = True
            nrot, ax0, ax1 = np.random.randint(0, 3), *np.random.permutation((0, 1, 2))[:2]
            rotation.append([nrot, ax0, ax1])

        def do_rotation(ar):
            for j in rotation:
                ar = np.rot90(ar, k=j[0], axes=(j[1], j[2]))
            return ar
        
        if rotate:
            nrotated +=1
            #Do for features
            features = []
            for i in range(nchannels):
                features.append(do_rotation(ftlist[i][x1:x2p, y1:y2p, z1:z2p]))#.copy()
            cube_features.append(np.stack(features, axis=-1))

            #Do for targets
            targets = []
            for i in range(ntarget):
                targets.append(do_rotation(targetlist[i][x1:x2, y1:y2, z1:z2]))#.copy()
            cube_target.append(np.stack(targets, axis=-1))
            
    print('Supplemented by rotation : ', nrotated)
    return cube_features, cube_target

    
def splitvoxels(ftlist, cube_size, shift=None, ncube=None):
    '''Split the meshes in ftlist in voxels of 'cube_size' in a regular fashion by
    shifting with 'shift' over the range of (0, ncp) on the mesh
    '''
    if type(ftlist) is not list: ftlist = [ftlist]
    ncp = ftlist[0].shape[0]
    if shift is None: shift = cube_size
    if ncube is None: ncube = int(ncp/shift)
    
    inp = []
    for i in range(ncube):
        for j in range(ncube):
            for k in range(ncube):
                x1, y1, z1 = i*shift, j*shift, k*shift
                x2, y2, z2 = x1+cube_size, y1+cube_size, z1+cube_size
                fts = np.stack([ar[x1:x2, y1:y2, z1:z2] for ar in ftlist], axis=-1)
                inp.append(fts)

    inp = np.stack(inp, axis=0)
    return inp



def readperiodic(ar, coords):
    '''
    '''
    def roll(ar, l1, l2, l0, axis):
        if l1<0 and l2>l0: 
            print('Inconsistency along axis %d'%axis)
            return None
        if l1<0: 
            ar=np.roll(ar, -l1, axis=axis)
            l1, l2 = 0, l2-l1
        elif l2>l0: 
            ar=np.roll(ar, l0-l2, axis=axis)
            l1, l2 = l1+l0-l2, l0
        return ar, l1, l2,

    if len(ar.shape) != len(coords): 
        print('dimensions inconsistent')
        return None
    ndim = len(coords)
    newcoords = []
    for i in range(ndim):
        ar, l1, l2 = roll(ar, coords[i][0], coords[i][1], ar.shape[i], i)
        newcoords.append([l1, l2])
    sl = []
    for i in range(ndim):
        sl.append(slice(*newcoords[i]))
    return ar[tuple(sl)]



def cubify(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    print(newshape, oldshape)
    repeats = (oldshape // newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)




def gethalomesh(bs, nc, seed, step=5, ncf=512, stepf=40, masswt=False, numd=1e-3, gridding='nn', path=None, getdata=False):

    if path is None: path = package_path + '/../../data/z00/'
    ftype = 'L%04d_N%04d_S%04d_%02dstep/'

    num = int(numd * bs**3)
    hposall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/PeakPosition/')[1:]    
    massall = tools.readbigfile(path + ftype%(bs, ncf, seed, stepf) + 'FOF/Mass/')[1:]    
    hposd = hposall[:num]
    massd = massall[:num].reshape(-1)*1e10
    if masswt: mass = massd
    else: mass = np.ones_like(massd)

    if gridding == 'nn': hmesh = tools.paintnn(hposd, bs, nc, mass=mass)
    else: hmesh = tools.paintcic(hposd, bs, nc, weights=mass)

    if getdata: return hmesh, hposd, massd
    else: return hmesh
            
    

def getgalmesh(bs, nc, seed, step=5, ncf=512, stepf=40, masswt=False,  gridding='nn', path=None):

    if path is None: path = package_path + '/../../data/z00/'
    ftype = 'L%04d_N%04d_S%04d_%02dstep/'

    hpath = path + ftype%(bs, ncf, seed, stepf) + 'galaxies_n05/galcat/'
    hposd = tools.readbigfile(hpath + 'Position/')
    massd = tools.readbigfile(hpath + 'Mass/').reshape(-1)*1e10
    galtype = tools.readbigfile(hpath + 'gal_type/').reshape(-1).astype(bool)
    if masswt: mass = massd
    else: mass = np.ones_like(massd)

    if gridding == 'nn':
        gmesh = tools.paintnn(hposd, bs, nc, mass=mass)
        satmesh = tools.paintnn(hposd[galtype], bs, nc, mass=mass[galtype])
        cenmesh = tools.paintnn(hposd[~galtype], bs, nc, mass=mass[~galtype])
    else:
        gmesh = tools.paintcic(hposd, bs, nc, mass=mass)
        satmesh = tools.paintcic(hposd[galtype], bs, nc, mass=mass[galtype])
        cenmesh = tools.paintcic(hposd[~galtype], bs, nc, mass=mass[~galtype])
    return cenmesh, satmesh, gmesh
