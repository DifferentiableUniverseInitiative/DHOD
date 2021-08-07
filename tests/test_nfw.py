import numpy as np
from numpy.testing import assert_allclose

from diffhod.distributions import RadialNFWProfile, NFWProfile
from halotools.empirical_models import NFWProfile as NFWProfile_ref


def test_nfw_mass_cdf():
  """
  Compares CDF values to halotools
  """
  model = NFWProfile_ref()
  scaled_radius = np.logspace(-2, 0, 100)

  for c in [5, 10, 20]:
    distr = RadialNFWProfile(concentration=c, Rvir=1)
    y = model.cumulative_mass_PDF(scaled_radius, conc=c)
    y_tf = distr.cdf(scaled_radius)
    assert_allclose(y, y_tf.numpy(), rtol=1e-4)


def test_nfw_mc_positions():
  """
  Compares samples with halotools and analytic density
  """
  model = NFWProfile_ref()

  for c in [5, 10, 20]:
    distr = RadialNFWProfile(concentration=c, Rvir=1)

    samples = model.mc_generate_nfw_radial_positions(
        num_pts=int(1e6), conc=c, halo_radius=1)
    samples_tf = distr.sample(1e6)

    h = np.histogram(samples, 32, density=True, range=[0.01, 1])
    h_tf = np.histogram(samples_tf, 32, density=True, range=[0.01, 1])
    x = 0.5 * (h[1][:-1] + h[1][1:])

    p = distr.prob(x)

    # Comparing histograms
    assert_allclose(h[0], h_tf[0], rtol=5e-2)
    # Comparing to prob
    assert_allclose(h_tf[0], p, rtol=5e-2)


def test_nfw_density():
  """
  Compares samples with halotools and analytic density
  """
  model = NFWProfile_ref()
  scaled_radius = np.logspace(-2, 0, 100)
  x = np.array([0,0,1]).reshape([1,3]) * scaled_radius.reshape([-1,1])
  for c in [5, 10, 20]:
    distr = NFWProfile(concentration=c, Rvir=1)
    y = model.dimensionless_mass_density(scaled_radius, conc=c)
    y_tf = distr.prob(x).numpy()
    assert_allclose(y/y[0], y_tf/y_tf[0], rtol=1e-4)
