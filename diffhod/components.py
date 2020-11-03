import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
tfd = tfp.distributions
tfb = tfp.bijectors

from diffhod.distributions import NFW


def Zheng07Cens(Mhalo, logMmin=ed.Deterministic(11.35, name='logMmin'), sigma_logM=ed.Deterministic(0.25, name='sigma_logM'), temperature=0.2, **kwargs):
    ''' expected number of central galaxies, <Ncen>, for Zheng+(2007) HOD model 

    <Ncen> = 0.5 * (1 + erf( (log Mh - log Mmin) / sigma_logM ))
    
    Parameters
    ----------
    Mhalo : float or array_like 
        Halo mass(es)

    logMmin : 
        Halo mass where the number of centrals in a halo transitions smoothly
        from 0 to 1.
    
    sigma_logM : 
        the scatter between stellar mass/luminosity and halo mass

    temperature : float 

    
    Returns
    -------
        
    '''
    Mhalo = tf.math.log(Mhalo) / tf.math.log(10.) # log10(Mhalo) 

    # Compute the mean number of centrals
    p = tf.clip_by_value(0.5 * (1+tf.math.erf((Mhalo - logMmin)/sigma_logM)), 
            1.e-4, 1-1.e-4) 
    return ed.RelaxedBernoulli(temperature, probs=p, name='zheng07Cens') 


def Zheng07SatsPoisson(Mhalo, Ncen, logM0=ed.Deterministic(11.2, name='logM0'), logM1=ed.Deterministic(12.4, name='logM1'), alpha=ed.Deterministic(0.83, name='alpha'), **kwargs):
    ''' Mean number of satellites, <Nsat>, for Zheng+(2007) HOD model. 

    <Nsat> = <Ncen> ((Mh - M0)/M1)^alpha 
    '''
    M0 = 10.**logM0
    M1 = 10.**logM1
    rate = Ncen.distribution.probs * ((Mhalo - M0)/M1)**alpha
    rate = tf.where(Mhalo < M0, 1e-4, rate)
    return ed.Poisson(rate=rate, name='zheng07Sats') 


def Zheng07SatsRelaxedBernoulli(Mhalo, Ncen, logM0=ed.Deterministic(11.2,
    name='logM0'), logM1=ed.Deterministic(12.4, name='logM1'),
    alpha=ed.Deterministic(0.83, name='alpha'), temperature=0.2,
    sample_shape=(100,), **kwargs):
    ''' Mean number of satellites, <Nsat>, for Zheng+(2007) HOD model. 

    <Nsat> = <Ncen> ((Mh - M0)/M1)^alpha 
    '''
    M0 = 10.**logM0
    M1 = 10.**logM1
    rate = Ncen.distribution.probs * (tf.nn.relu(Mhalo - M0)/M1)**alpha
    return ed.RelaxedBernoulli(temperature=temperature,
            probs=tf.clip_by_value(rate/sample_shape[0],1.e-5,1-1e-4),
            sample_shape=sample_shape, name='zheng07Sats')


def NFWProfile(pos, concentration, Rvir, sample_shape, **kwargs):
    '''
    '''
    pos = ed.as_random_variable(tfd.TransformedDistribution(distribution=tfd.VonMisesFisher(tf.one_hot(tf.zeros_like(concentration, dtype=tf.int32),3), 0), bijector=tfb.AffineScalar(shift=pos, scale=tf.expand_dims(ed.as_random_variable(NFW(concentration, Rvir, name='radius'), sample_shape=sample_shape), axis=-1)), name='position'), sample_shape=sample_shape)
    return pos
