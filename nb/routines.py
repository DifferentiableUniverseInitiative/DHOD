import tensorflow as tf
## Should wrap this into a class...
import numpy as np

class Power_Spectrum():

    def __init__(self,shape,boxsize,kmin,dk):
        self.shape = shape
        self.boxsize = boxsize
        self.kmin = kmin
        self.dk = dk
        
        self._initialize_pk()
        
    def _initialize_pk(self):
        """
           Helper function to initialize various (fixed) values for powerspectra... not differentiable!
        """
        I = np.eye(len(self.shape), dtype='int') * -2 + 1

        W = np.empty(self.shape, dtype='f4')
        W[...] = 2.0
        W[..., 0] = 1.0
        W[..., -1] = 1.0

        kmax = np.pi * np.min(self.shape)/np.max(self.boxsize) + self.dk/2
        self.kedges = np.arange(self.kmin, kmax, self.dk)

        self.k = [np.fft.fftfreq(N, 1. / (N * 2 * np.pi / L))[:pkshape].reshape(kshape) for N, L, kshape, pkshape in zip(self.shape, self.boxsize, I, self.shape)]
        kmag = sum(ki ** 2 for ki in self.k) ** 0.5

        xsum = np.zeros(len(self.kedges) + 1)
        Nsum = np.zeros(len(self.kedges) + 1)

        dig = np.digitize(kmag.flat, self.kedges)

        xsum.flat += np.bincount(dig, weights=(W * kmag).flat, minlength=xsum.size)
        Nsum.flat += np.bincount(dig, weights=W.flat, minlength=xsum.size)
        self.dig = dig
        self.Nsum = Nsum
        self.W = W
        self.xsum = xsum
      #  return dig, Nsum, xsum, W, k, kedges

    @tf.function
    def pk_tf(self,field):  
        """
        Calculate the powerspectra given real space field

        Args:

            field: real valued field 
            kmin: minimum k-value for binned powerspectra
            dk: differential in each kbin
            shape: shape of field to calculate field (can be strangely shaped?)
            boxsize: length of each boxlength (can be strangly shaped?)

        Returns:

            kbins: the central value of the bins for plotting
            power: real valued array of power in each bin

        """

        batchsize = field.shape[0] #batch size
        nc = field.shape[1]
        #print(np.sum(bs))
        #dig, Nsum, xsum, W, k, kedges = _initialize_pk(shape,boxsize,kmin,dk)

        #convert field to complex for fft
        field_complex = tf.dtypes.cast(field,dtype=tf.complex64)

        #fast fourier transform
        fft_image = tf.map_fn(tf.signal.fft3d, field_complex)#, dtype=None, parallel_iterations=None, back_prop=True,
        #swap_memory=False, infer_shape=True, name=None
        #)


        #absolute value of fast fourier transform
        pk = tf.math.real(fft_image*tf.math.conj(fft_image))
        #calculating powerspectra
        Psum = tf.zeros(tf.size(self.kedges) + 1, dtype=tf.complex64)
        real = tf.reshape(tf.math.real(pk),[batchsize,-1,])
        imag = tf.reshape(tf.math.imag(pk),[batchsize,-1,])


        #def bincount func
        @tf.function
        def bincount(x):
            return tf.math.bincount(self.dig, weights=(tf.reshape(self.W,[-1])  * x), minlength=tf.size(self.xsum))
        #Psum1 = tf.dtypes.cast(tf.vectorized_map(bincount, imag),dtype=tf.complex64)*1j
        #Psum2 = tf.dtypes.cast(tf.vectorized_map(bincount, real),dtype=tf.complex64)
        Psum1 = tf.dtypes.cast(tf.map_fn(
         bincount, imag, dtype=None, parallel_iterations=None, back_prop=True,
         swap_memory=False, infer_shape=True, name=None
         ),dtype=tf.complex64)*1j

        Psum2 = tf.dtypes.cast(tf.map_fn(
         bincount, real, dtype=None, parallel_iterations=None, back_prop=True,
         swap_memory=False, infer_shape=True, name=None
         ),dtype=tf.complex64)


        power = ((Psum + Psum1+Psum2)/ self.Nsum)[:,1:-1]* tf.cast(self.boxsize[0]**3,dtype=tf.complex64)

        #normalization for powerspectra
        norm = tf.dtypes.cast(nc**3,dtype=tf.float32)**2

        #find central values of each bin
        kbins = self.kedges[:-1]+ (self.kedges[1:] - self.kedges[:-1])/2

        return kbins,tf.dtypes.cast(power,dtype=tf.float32)/norm
    
#TF2 compatable painter
@tf.function#(experimental_relax_shapes=True)
def cic_paint(mesh, part, weight=None, name="CiCPaint"):
  """
  Paints particules on a 3D mesh.
  Parameters:
  -----------
  mesh: tensor (batch_size, nc, nc, nc)
    Input 3D mesh tensor
  part: tensor (batch_size, npart, 3)
    List of 3D particle coordinates, assumed to be in mesh units if
    boxsize is None
  weight: tensor (batch_size, npart)
    List of weights  for each particle
  """
  with tf.name_scope(name):
    mesh = tf.convert_to_tensor(mesh, name="mesh")
    part = tf.convert_to_tensor(part, name="part")
    if weight is not None:
      weight = tf.convert_to_tensor(weight, name="weight")

    shape = tf.shape(mesh)
    batch_size, nx, ny, nz = shape[0], shape[1], shape[2], shape[3]
    nc = nz

    # Flatten part if it's not already done
    if len(part.shape) > 3:
      part = tf.reshape(part, (batch_size, -1, 3))

    # Extract the indices of all the mesh points affected by each particles
    part = tf.expand_dims(part, 2)
    floor = tf.floor(part)
    connection = tf.expand_dims(tf.constant([[[0, 0, 0], [1., 0, 0],[0., 1, 0],
                                              [0., 0, 1],[1., 1, 0],[1., 0, 1],
                                              [0., 1, 1],[1., 1, 1]]]), 0)

    neighboor_coords = floor + connection
    kernel = 1. - tf.abs(part - neighboor_coords)
    # Replacing the reduce_prod op by manual multiplication
    # TODO: figure out why reduce_prod was crashing the Hessian computation
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    if weight is not None: kernel = tf.multiply(tf.expand_dims(weight, axis=-1) , kernel)

    neighboor_coords = tf.cast(neighboor_coords, tf.int32)
    neighboor_coords = tf.math.mod(neighboor_coords , nc)

    # Adding batch dimension to the neighboor coordinates
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
    b = tf.tile(batch_idx, [1] + list(neighboor_coords.get_shape()[1:-1]) + [1])
    neighboor_coords = tf.concat([b, neighboor_coords], axis=-1)

    update = tf.scatter_nd(tf.reshape(neighboor_coords, (-1, 8,4)),tf.reshape(kernel, (-1, 8)),[batch_size, nx, ny, nz])
    mesh = mesh + update
    return mesh


@tf.function
def paint_galaxies(gal_cat, nc=128):
    # Take centrals and rescale them to the boxsize
    
    bs = gal_cat['n_sat'].shape[1]
    
    sample1 = gal_cat['pos_cen'] / 128. * nc
    weights1 = gal_cat['n_cen']
    # Take sats and rescale them to the boxize
  
    sample2 = tf.reshape(gal_cat['pos_sat'], [-1,3]) / 128. * nc
    weights2 = tf.reshape(tf.transpose(gal_cat['n_sat'],[1,0,2]),[bs,-1])
    
    sample1_r = tf.tile(tf.expand_dims(sample1,0),[bs,1,1])
    print(sample1_r.shape,weights1.shape)
    rho1 = cic_paint(tf.zeros((bs, nc, nc, nc)),sample1_r, weights1)
    sample2_r = tf.tile(tf.expand_dims(sample2,0),[bs,1,1])
    print(sample2_r.shape,weights2.shape)

    rho2 = cic_paint(tf.zeros((bs, nc, nc, nc)),sample2_r, weights2)
    rho = rho1+rho2
    return rho

# sampling galaxies from the model, with given params
@tf.function
def sample(halo_cat, logMmin, sigma_logM, logM0, logM1, alpha):
    return paint_galaxies(hod(halo_cat,logMmin, sigma_logM, logM0, logM1, alpha))



### HOD 

import tensorflow as tf
from tensorflow_probability import edward2 as ed
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from diffhod.distributions import NFW

def Zheng07Cens(halo_mvir,
                logMmin=ed.Deterministic(11.35, name='logMmin'),
                sigma_logM=ed.Deterministic(0.25, name='sigma_logM'),
                temperature=0.2,
                name='zheng07Cens', **kwargs):
  halo_mvir = tf.math.log(halo_mvir) / tf.math.log(10.)
  # Compute the mean number of centrals
  p = tf.clip_by_value(0.5 * (1+tf.math.erf((halo_mvir - tf.reshape(logMmin,(-1,1)))/tf.reshape(sigma_logM,(-1,1)))), 1.e-4, 1-1.e-4)
  return ed.RelaxedBernoulli(temperature, probs=p, name=name)

def Zheng07SatsPoisson(halo_mvir,
                n_cen,
                logM0=ed.Deterministic(11.2, name='logM0'),
                logM1=ed.Deterministic(12.4, name='logM1'),
                alpha=ed.Deterministic(0.83, name='alpha'),
                name='zheng07Sats', **kwargs):
  M0 = tf.pow(10.,logM0)
  M1 = tf.pow(10.,logM1)
  rate = n_cen.distribution.probs * tf.math.pow((halo_mvir - tf.reshape(M0,(-1,1)))/(tf.reshape(M1,(-1,1))),tf.reshape(alpha,(-1,1)))
  rate = tf.where(halo_mvir < tf.reshape(M0,(-1,1)), 1e-4, rate)
  return ed.Poisson(rate=rate, name=name)

def Zheng07SatsRelaxedBernoulli(halo_mvir,
        n_cen,
        sample_shape,
        logM0=ed.Deterministic(11.2, name='logM0'),
        logM1=ed.Deterministic(12.4, name='logM1'),
        alpha=ed.Deterministic(0.83, name='alpha'),
        temperature=0.2,
        name='zheng07Sats', **kwargs):
    M0 = tf.pow(10.,logM0)
    M1 = tf.pow(10.,logM1)
    print(M0)
    
    num = halo_mvir - tf.reshape(M0,(-1,1))
    
    rate = n_cen.distribution.probs * tf.pow(tf.nn.relu(num/tf.reshape(M1,(-1,1))),tf.reshape(alpha,(-1,1)))
    return ed.RelaxedBernoulli(temperature=temperature,
                             probs=tf.clip_by_value(rate/sample_shape[0],1.e-5,1-1e-4),
                             sample_shape=sample_shape)


def NFWProfile(pos,
               concentration,
               Rvir,
               sample_shape, **kwargs):

  pos = ed.as_random_variable(tfd.TransformedDistribution(distribution=tfd.VonMisesFisher(tf.one_hot(tf.zeros_like(concentration, dtype=tf.int32),3), 0),
                                   bijector=tfb.AffineScalar(shift=pos, scale=tf.expand_dims(ed.as_random_variable(NFW(concentration, Rvir, name='radius'), sample_shape=sample_shape), axis=-1)),
                                                        name='position'), sample_shape=sample_shape)
  return pos



@tf.function
def hod(halo_cat, logMmin, sigma_logM, logM0, logM1, alpha, max_sat=34, temp=0.01,bs=10):
  ### Occupation model ###
  n_cen = Zheng07Cens(halo_cat['halo_mvir'],
                      sigma_logM=sigma_logM,
                      logMmin=logMmin,
                      temperature=temp)
  n_sat = Zheng07SatsRelaxedBernoulli(halo_cat['halo_mvir'],
                                      n_cen,
                                      logM0=logM0,
                                      logM1=logM1,
                                      alpha=alpha,
                                      sample_shape=(max_sat,),
                                      temperature=temp)
  
  ### Phase Space model ###
  # Centrals are just located at center of halo
  pos_cen = ed.Deterministic(tf.stack([halo_cat['halo_x'],
                                        halo_cat['halo_y'],
                                        halo_cat['halo_z']], axis=-1))

  # Satellites follow an NFW profile centered on halos
  pos_sat = NFWProfile(pos=pos_cen,
                        concentration=halo_cat['halo_nfw_conc'],
                        Rvir=halo_cat['halo_rvir'],
                        sample_shape=(max_sat,))
  
  return {'pos_cen':pos_cen,'n_cen':n_cen, 'pos_sat':pos_sat,  'n_sat':n_sat}