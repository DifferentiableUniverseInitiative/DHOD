from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf


from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions.internal import slicing
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import name_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


class NFW(distribution.Distribution):
  r"""NFW radial distribution.

  This distribution is useful to sample satelite galaxies according to an NFW
  radial profile.

  TODO: add math
  """

  def __init__(self,
               concentration,
               Mvir,
               Rvir,
               validate_args=False,
               allow_nan_stats=True,
               name="nfw"):
    """Construct an NFW profile with specified concentration, virial mass and
    radius.

    Args:
      concentration: Floating point tensor; concentration of the NFW profile
      Mvir: Floating point tensor; the virial mass of the profile
        Must contain only positive values.
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
      dtype = dtype_util.common_dtype([concentration, Mvir, Rvir], dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          value=concentration, name='concentration', dtype=dtype)
      self._Mvir = tensor_util.convert_nonref_to_tensor(
          value=Mvir, name='Mvir', dtype=dtype)
      self._Rvir = tensor_util.convert_nonref_to_tensor(
          value=Rvir, name='Rvir', dtype=dtype)

    super(NFW, self).__init__(
        dtype=self._concentration.dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name)

  @property
  def concentration(self):
    """Distribution parameter for the concentration."""
    return self._concentration

  @property
  def Mvir(self):
    """Distribution parameter for the virial mass."""
    return self._Mvir

  @property
  def Rvir(self):
    """Distribution parameter for the virial radius."""
    return self._Rvir

  def _q(self, r):
    """Standardize input radius `r` to a standard `q`"""
    with tf.name_scope('standardize'):
      return r/self.Rvir

  def _prob_unormalized(self, q):
    """Returns the unormalized pdf"""
    x = q * self.concentration
    retun tf.log(1.0 + y) - y / (1.0 + y)

  def _log_prob(self, r):
    q = self._q(r)
    p = self._prob_unormalized(q) / self._prob_unormalized(1.0)
    return tf.log(p)
