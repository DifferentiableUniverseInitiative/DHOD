# This distribution is a temporary fix until a permanent solution gets merged in TFP
# See this issue https://github.com/tensorflow/probability/issues/1393
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class CustomSigmoid(tfb.Sigmoid):
  """ Custom bijector to bypass issue in TFP sigmoid gradients
  """

  def _forward(self, x):
    if self._is_standard_sigmoid:
      return tf.sigmoid(x)
    return self.high * tf.sigmoid(x) + self.low * tf.sigmoid(-x)


class RelaxedBernoulli(tfd.RelaxedBernoulli):
  """ Custom Relaxed Bernoulli using the above sigmoid bijector
  """

  def _transformed_logistic(self):
    logistic_scale = tf.math.reciprocal(self._temperature)
    logits_parameter = self._logits_parameter_no_checks()
    logistic_loc = logits_parameter * logistic_scale
    return tfd.TransformedDistribution(
        distribution=tfd.Logistic(
            logistic_loc, logistic_scale, allow_nan_stats=self.allow_nan_stats),
        bijector=CustomSigmoid())
