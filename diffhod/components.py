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
  rate = n_cen.distribution.probs * ((halo_mvir - tf.reshape(M0,(-1,1)))/tf.reshape(M1,(-1,1)))**tf.reshape(alpha,(-1,1))
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
    rate = n_cen.distribution.probs * (tf.nn.relu(halo_mvir - tf.reshape(M0,(-1,1)))/tf.reshape(M1,(-1,1)))**tf.reshape(alpha,(-1,1))

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
