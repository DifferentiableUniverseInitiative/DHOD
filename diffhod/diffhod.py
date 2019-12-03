'''
'''
import tensorflow as tf
import tensorflow_probability as tfp


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


def hod(Mhalo, theta, temperature=0.2, model='zheng07'):
    if model == 'zheng07': 
        # sample relaxed bernoulli dist. for centrals
        bern = tfp.distributions.RelaxedBernoulli(temperature, probs=Ncen(Mhalo, theta, model=model)).sample()

        # sample poisson distribution estimated by N(mu, mu) 
        # THIS IS VERY APPROXIMATE AND NOT CORRECT 
        # REPLACE WITH SOMETHING BETTER
        nsat = Nsat(Mhalo, theta, model=model)
        pois = tfp.distributions.Normal(loc=nsat, scale=nsat).sample()
        return bern + pois
