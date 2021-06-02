'''
'''
import tensorflow as tf
import tensorflow_probability as tfp
from .components import *


@tf.function
def hod(halo_cat, logMmin, sigma_logM, logM0, logM1, alpha, max_sat=20, temp=0.01,bs=10):
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


## Old routines?
def Ncen(Mhalo, theta, model='zheng07'):
    ''' mean central occupation 
    '''
    if model == 'zheng07':
        Mmin, siglogm = theta[0], theta[1] 
        return tf.clip_by_value(0.5 * (1+tf.math.erf((Mhalo - Mmin)/siglogm)), 1.e-4, 1-1.e-4)
    else: 
        raise NotImplementedError 
        
def Nsat(Mhalo, theta, model='zheng07'):
    ''' mean satellite occupation 
    '''
    if model =='zheng07':
        Mmin, siglogm, M0, M1, alpha = theta
        return tf.clip_by_value(Ncen(Mhalo, theta, model=model) * (tf.clip_by_value((Mhalo - M0)/M1, 1e-4, 1e4))**alpha, 1.e-4, 1e4)
    else:
        raise NotImplementedError


def __hod(Mhalo, theta, temperature=0.2, model='zheng07'):
    if model == 'zheng07': 
        # sample relaxed bernoulli dist. for centrals
        cens = tfp.distributions.RelaxedBernoulli(temperature, probs=Ncen(Mhalo, theta, model=model))

        # sample poisson distribution 
        sats = tfp.distributions.Poisson(rate=Nsat(Mhalo, theta, model=model))
        return cens.sample(), tf.stop_gradient(sats.sample() - sats.rate) + sats.rate
    

