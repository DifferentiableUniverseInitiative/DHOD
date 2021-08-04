# Tests the components of the zheng07 model
import numpy as np
from numpy.testing import assert_allclose

from halotools.empirical_models import PrebuiltHodModelFactory
from diffhod.components import Zheng07Cens, Zheng07SatsPoisson


def test_zheng07_mean_occupation_centrals():
  """
  Compares the central probability with halotools
  """
  zheng07_model = PrebuiltHodModelFactory('zheng07')

  test_mass = np.logspace(10, 15).astype('float32')
  p = zheng07_model.mean_occupation_centrals(prim_haloprop=test_mass)
  p_tf = Zheng07Cens(test_mass,
                     **(zheng07_model.param_dict)).distribution.probs
  assert_allclose(p, p_tf.numpy(), atol=1.e-3)


def test_zheng07_mean_occupation_satellites():
  """
  Compare the poisson rate with halotools
  """
  zheng07_model = PrebuiltHodModelFactory('zheng07', modulate_with_cenocc=True)

  test_mass = np.logspace(10, 15).astype('float32')
  p = zheng07_model.mean_occupation_satellites(prim_haloprop=test_mass)

  n_cen = Zheng07Cens(test_mass, **(zheng07_model.param_dict))
  n_sat = Zheng07SatsPoisson(test_mass, n_cen, **(zheng07_model.param_dict))

  assert_allclose(p, n_sat.distribution.rate.numpy(), atol=1.e-2)


def test_zheng07_default_params():
  """
  Compare the that we are using the same default parameters as halotools
  """
  zheng07_model = PrebuiltHodModelFactory('zheng07', modulate_with_cenocc=True)

  test_mass = np.logspace(10, 15).astype('float32')
  p = zheng07_model.mean_occupation_satellites(prim_haloprop=test_mass)

  n_cen = Zheng07Cens(test_mass)
  n_sat = Zheng07SatsPoisson(test_mass, n_cen)

  assert_allclose(p, n_sat.distribution.rate.numpy(), atol=1.e-2)
