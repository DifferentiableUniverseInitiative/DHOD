from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import lambertw
from tensorflow_probability.python.internal import samplers

__all__ = [
    'RadialNFWProfile',
    'NFWProfile'
]

class RadialNFWProfile(distribution.Distribution):
  r"""NFW radial mass distribution.

  This distribution is useful to sample satelite galaxies according to an NFW
  radial profile.

  Implementation found in this class follows: https://arxiv.org/abs/1805.09550
  """

  def __init__(self,
               concentration,
               Rvir,
               validate_args=False,
               allow_nan_stats=True,
               name="nfw"):
    """Construct an NFW profile with specified concentration, virial radius.

    Args:
      concentration: Floating point tensor; concentration of the NFW profile
      Rvir: Floating point tensor; the virial radius of the profile
        Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs. Default value: `False` (i.e., do not validate args).
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or more
        of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'nfw'.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([concentration, Rvir],
                                      dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          value=concentration, name='concentration', dtype=dtype)
      self._Rvir = tensor_util.convert_nonref_to_tensor(
          value=Rvir, name='Rvir', dtype=dtype)

    super(RadialNFWProfile, self).__init__(
        dtype=self._concentration.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('concentration', 'Rvir'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(concentration=0, Rvir=0)

  @property
  def concentration(self):
    """Distribution parameter for the concentration."""
    return self._concentration

  @property
  def Rvir(self):
    """Distribution parameter for the virial radius."""
    return self._Rvir

  def _batch_shape_tensor(self, concentration=None, Rvir=None):
    return prefer_static.broadcast_shape(
        prefer_static.shape(
            self.concentration if concentration is None else concentration),
        prefer_static.shape(self.Rvir if Rvir is None else Rvir))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.concentration.shape, self.Rvir.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _q(self, r):
    """Standardize input radius `r` to a standard `q`"""
    with tf.name_scope('standardize'):
      Rvir = tf.convert_to_tensor(self.Rvir)
      return r / Rvir

  def _prob(self, r):
    concentration = tf.convert_to_tensor(self.concentration)
    q = self._q(r)
    p = (q * concentration**2) / (
        ((q * concentration) + 1.0)**2 *
        (1.0 / (concentration + 1.0) + tf.math.log(concentration + 1.0) - 1.0))
    return p

  def _cdf_unormalized(self, q):
    """Returns the unormalized cdf"""
    with tf.name_scope('prob_unormalized'):
      concentration = tf.convert_to_tensor(self.concentration)
      x = q * self.concentration
      return tf.math.log(1.0 + x) - x / (1.0 + x)

  def _cdf(self, r):
    q = self._q(r)
    p = self._cdf_unormalized(q) / self._cdf_unormalized(1.0)
    return p

  def _log_cdf(self, r):
    q = self._q(r)
    return tf.math.log(self._cdf_unormalized(q)) - tf.math.log(
        self._cdf_unormalized(1.0))

  def _quantile(self, p):
    """ Inverse CDF aka quantile function of the NFW profile.
    Returns normalized q radius.
    """
    with tf.name_scope('quantile'):
      concentration = tf.convert_to_tensor(self.concentration)
      #TODO: add checks that 0<= p <=1
      p *= self._cdf_unormalized(1.0)
      q = (-(1. / lambertw(-tf.exp(-p - 1))) - 1)
      return q / concentration

  def _sample_n(self, n, seed=None):
    Rvir = tf.convert_to_tensor(self.Rvir)
    concentration = tf.convert_to_tensor(self.concentration)
    shape = tf.concat(
        [[n],
         self._batch_shape_tensor(concentration=concentration, Rvir=Rvir)], 0)
    # Sample from uniform distribution
    dt = dtype_util.as_numpy_dtype(self.dtype)
    uniform_samples = samplers.uniform(
        shape=shape,
        minval=np.nextafter(dt(0.), dt(1.)),
        maxval=1.,
        dtype=self.dtype,
        seed=seed)

    # Transform using the quantile function
    return self._quantile(uniform_samples) * Rvir


class NFWProfile(RadialNFWProfile):
  r""" 3D NFW mass distribution.
  
  This distribution is useful to sample satelite galaxies according to an NFW
  profile.

  Implementation found in this class follows: https://arxiv.org/abs/1805.09550
  """

  def _event_shape_tensor(self):
    return tf.constant([3], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([3])

  def _prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    r = tf.norm(x, axis=-1)
    q = self._q(r)
    qc = q * concentration
    # TODO: properly normalize the density
    return 1. / (qc * (1 + qc)**2)

  def _cdf(self, x):
    raise NotImplementedError
    
  def _log_cdf(self, x):
    raise NotImplementedError
    
  def _sample_n(self, n, seed):
    radial_seed, otherdims_seed = samplers.split_seed(seed,
                                                     salt='NFW')
    Rvir = tf.convert_to_tensor(self.Rvir)
    concentration = tf.convert_to_tensor(self.concentration)

    r = super()._sample_n(n, seed=radial_seed)
    shape = tf.concat(
        [[n],
        self._batch_shape_tensor(concentration=concentration, Rvir=Rvir),
        self._event_shape_tensor()], 0)

    x = tf.math.l2_normalize(
        samplers.normal(
            shape, seed=otherdims_seed, dtype=self.dtype),
        axis=-1)
    return x * tf.expand_dims(r, axis=-1)
