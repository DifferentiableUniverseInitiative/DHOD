import tensorflow as tf
import edward2 as ed
import tensorflow_probability as tfp
from diffhod.distributions import NFW

tfd = tfp.distributions
tfb = tfp.bijectors


def Zheng07Cens(Mhalo,
                logMmin=ed.Deterministic(12.02, name='logMmin'),
                sigma_logM=ed.Deterministic(0.26, name='sigma_logM'),
                temperature=0.02,
                name='zheng07Cens',
                **kwargs):
  """ Expected number of central galaxies, <Ncen>, in a halo of mass Mhalo
  for Zheng+(2007) HOD model:

    <Ncen> = 0.5 * (1 + erf( (log Mh - log Mmin) / sigma_logM ))
  
  Args:
    Mhalo: float or array_like; mass of the halo.
    logMmin: float; Halo mass where the number of centrals in a halo transitions smoothly
      from 0 to 1.
    sigma_logM: float; the scatter between stellar mass/luminosity and halo mass
    temperature : float; temperature for the Relaxed Bernoulli distribution

  Returns:
    ncentral. RandomVariable: Number of central galaxy in the halo.
  """
  Mhalo = tf.math.log(Mhalo) / tf.math.log(10.)  # log10(Mhalo)

  # Compute the mean number of centrals
  p = tf.clip_by_value(0.5 * (1 + tf.math.erf((Mhalo - logMmin) / sigma_logM)),
                       1.e-4, 1 - 1.e-4)

  return ed.RelaxedBernoulli(temperature, probs=p, name=name)


def _Zheng07SatsRate(Mhalo, logM0, logM1, alpha):
  M0 = tf.pow(10., logM0)
  M1 = tf.pow(10., logM1)
  return tf.math.pow(tf.nn.relu(Mhalo - M0) / M1, alpha)


def Zheng07SatsPoisson(Mhalo,
                       Ncen,
                       logM0=ed.Deterministic(11.38, name='logM0'),
                       logM1=ed.Deterministic(13.31, name='logM1'),
                       alpha=ed.Deterministic(1.06, name='alpha'),
                       name='zheng07Sats',
                       **kwargs):
  ''' Mean number of satellites, <Nsat>, for Zheng+(2007) HOD model. 

    <Nsat> = <Ncen> ((Mh - M0)/M1)^alpha 
  '''
  M0 = tf.pow(10., logM0)
  rate = Ncen.distribution.probs * _Zheng07SatsRate(Mhalo, logM0, logM1, alpha)
  rate = tf.where(Mhalo < M0, 1e-4, rate)
  return ed.Poisson(rate=rate, name=name)


def Zheng07SatsRelaxedBernoulli(Mhalo,
                                Ncen,
                                logM0=ed.Deterministic(11.38, name='logM0'),
                                logM1=ed.Deterministic(13.31, name='logM1'),
                                alpha=ed.Deterministic(1.06, name='alpha'),
                                temperature=0.02,
                                sample_shape=(100, ),
                                name='zheng07Sats',
                                **kwargs):
  """ Expected number of satellite galaxies, <Nsat>, in a halo of mass Mhalo
  for Zheng+(2007) HOD model:

    <Nsat> = <Ncen> ((Mh - M0)/M1)^alpha 
  
  Args:
    Mhalo: float or array_like; mass of the halo.
    Ncen: floar or array_like; number of centrals.
    logM0, logM1, alpha: float; Parameters of HOD model
    temperature : float; temperature for the Relaxed Bernoulli distribution
    sample_shape: maximum number of satellite per halo.

  Returns:
    nsat. array_like RandomVariable: Tensor of shape `sample_shape` with binary entries
  """
  rate = Ncen.distribution.probs * _Zheng07SatsRate(Mhalo, logM0, logM1, alpha)
  return ed.RelaxedBernoulli(temperature=temperature,
                             probs=tf.clip_by_value(rate / sample_shape[0],
                                                    1.e-5, 1 - 1e-4),
                             sample_shape=sample_shape,
                             name=name)


def NFWProfile(center,
               concentration,
               Rvir,
               sample_shape,
               name='positions',
               **kwargs):
  """ Cartesian coordinates drawn from isotropic NFW profile.

  Args:
    center: Tensor of shape [..., 3]; halo center.
    concentration: float; concentration parameter for NFW profile
    Rvir: float; Virial radius of halo
    sample_shape: maximum number of satellite per halo.
  
  Returns:
    positions. array_like RandomVariable of cartesian coordinates.
  """
  pos = ed.RandomVariable(tfd.TransformedDistribution(
      distribution=tfd.VonMisesFisher(
          tf.one_hot(tf.zeros_like(concentration, dtype=tf.int32), 3), 0),
      bijector=tfb.Shift(center)(tfb.Scale(
          tf.expand_dims(ed.RandomVariable(NFW(concentration,
                                               Rvir,
                                               name='radius'),
                                           sample_shape=sample_shape),
                         axis=-1))),
      name='position'),
                          sample_shape=sample_shape,
                          name=name)
  return pos
