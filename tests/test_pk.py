import os

#specify on CPU for debug...
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf

from numpy.testing import assert_allclose

from diffhod.mock_observables import pk

fid_power = 2.4 * np.ones(5)


def test_pk():
  field = tf.random.normal([1, 100, 100, 100])
  pk_calculator = pk.Power_Spectrum(shape=[100, 100, 100],
                                    boxsize=[128, 128, 128],
                                    kmin=0.001,
                                    dk=0.5)
  power = pk_calculator.pk_tf(field)
  assert_allclose(power[1][0], fid_power, rtol=2e-1)
