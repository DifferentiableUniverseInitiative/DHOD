import os

#specify on CPU for debug...
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf

from numpy.testing import assert_allclose

from diffhod.mock_observables import pk

fid_power = 2.4*np.ones(5)

def test_pk():
    field = tf.random.normal([100,100,100])
    power = pk.pk(field,shape = field.shape, boxsize= np.array([128,128,128]),kmin=0.001)
    assert_allclose(power[1], fid_power, rtol=2e-1)
