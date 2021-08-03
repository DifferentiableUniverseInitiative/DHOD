import numpy as np
import numpy

import tensorflow as tf


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
    