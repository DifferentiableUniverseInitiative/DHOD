from diffhod.distributions.RelaxedBernoulli import RelaxedBernoulli, CustomSigmoid
import numpy as np
import tensorflow as tf
import edward2 as ed
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# Defining Edward random variables for our custom distributions
import diffhod.distributions as custom_distributions

NFW = ed.make_random_variable(custom_distributions.NFWProfile)

def RelaxedBernoulli(temperature, probs, sample_shape=(), name='bernoulli'):
  logistic_scale = tf.math.reciprocal(temperature)
  logits_parameter = tf.math.log(probs) - tf.math.log1p(-probs)
  logistic_loc = logits_parameter * logistic_scale
  l = ed.Logistic(logistic_loc*0, logistic_scale*0+1, name=name+'_Logistic',
                  sample_shape=sample_shape)
  bij = tfb.Chain([CustomSigmoid(), tfb.Shift(logistic_loc), tfb.Scale(logistic_scale)])
  x = bij(l)
  return x

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
  with tf.name_scope('Zheng07Cens'):
    Mhalo = tf.convert_to_tensor(Mhalo)
    logMmin = tf.convert_to_tensor(logMmin)
    sigma_logM = tf.convert_to_tensor(sigma_logM)
    # Test for batched parameters
    shape_params = logMmin.get_shape()
    if len(shape_params) > 0:
      logMmin = tf.reshape(logMmin, [-1, 1])
      sigma_logM = tf.reshape(sigma_logM, [-1, 1])

    Mhalo = tf.math.log(Mhalo) / np.log(10.)  # log10(Mhalo)

    # Compute the mean number of centrals
    p = tf.clip_by_value(
        0.5 * (1 + tf.math.erf((Mhalo - logMmin) / sigma_logM)), 1.e-4,
        1 - 1.e-4)

    return RelaxedBernoulli(temperature, probs=p, name=name)


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
  with tf.name_scope('Zheng07Sats'):
    Mhalo = tf.convert_to_tensor(Mhalo)
    logM0 = tf.convert_to_tensor(logM0)
    logM1 = tf.convert_to_tensor(logM1)
    alpha = tf.convert_to_tensor(alpha)
    # Test for batched parameters
    shape_params = logM0.get_shape()
    if len(shape_params) > 0:
      logM0 = tf.expand_dims(logM0, -1)
      logM1 = tf.expand_dims(logM1, -1)
      alpha = tf.expand_dims(alpha, -1)

    M0 = tf.pow(10., logM0)
    rate = Ncen.distribution.probs * _Zheng07SatsRate(Mhalo, logM0, logM1,
                                                      alpha)
    rate = tf.where(Mhalo < M0, 1e-4, rate)
    return ed.Poisson(rate=rate, name=name)


def Zheng07SatsRelaxedBernoulli(Mhalo,
                                Ncen,
                                logM0=ed.Deterministic(11.38, name='logM0'),
                                logM1=ed.Deterministic(13.31, name='logM1'),
                                alpha=ed.Deterministic(1.06, name='alpha'),
                                temperature=0.02,
                                sample_shape=(100,),
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
  with tf.name_scope('Zheng07Sats'):
    Mhalo = tf.convert_to_tensor(Mhalo)
    logM0 = tf.convert_to_tensor(logM0)
    logM1 = tf.convert_to_tensor(logM1)
    alpha = tf.convert_to_tensor(alpha)
    # Test for batched parameters
    shape_params = logM0.get_shape()
    if len(shape_params) > 0:
      logM0 = tf.expand_dims(logM0, -1)
      logM1 = tf.expand_dims(logM1, -1)
      alpha = tf.expand_dims(alpha, -1)

    rate =  _Zheng07SatsRate(Mhalo, logM0, logM1,
                                                      alpha)
    return Ncen * RelaxedBernoulli(
        temperature=temperature,
        probs=tf.clip_by_value(rate / sample_shape[0], 1.e-5, 1 - 1e-4),
        sample_shape=sample_shape,
        name=name)


# By default, we will refer to the Bernoulli model as Zheng07Sats
Zheng07Sats = Zheng07SatsRelaxedBernoulli


def NFWProfile(pos,
               concentration,
               Rvir,
               sample_shape=(100,),
               name='positions',
               **kwargs):
  """ Cartesian coordinates drawn from isotropic NFW profile.

  Args:
    pos: Tensor of shape [..., 3]; halo center.
    concentration: float; concentration parameter for NFW profile
    Rvir: float; Virial radius of halo
    sample_shape: maximum number of satellite per halo.
  
  Returns:
    positions. array_like RandomVariable of cartesian coordinates.
  """
  with tf.name_scope('NFWProfile'):
    pos = tf.convert_to_tensor(pos)
    concentration = tf.convert_to_tensor(concentration)
    Rvir = tf.convert_to_tensor(Rvir)
    # Sample a standard NFW and then rescale it
    positions = NFW(concentration=tf.ones_like(concentration), 
               Rvir=tf.ones_like(Rvir), 
               name=name,
               sample_shape=sample_shape)
    factor = tf.expand_dims(tf.expand_dims(Rvir / concentration,-1),0)
    return positions * factor + pos