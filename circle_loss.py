import tensorflow as tf
import numpy as np
k = tf.keras
kl = tf.keras.layers
kls = tf.keras.losses
kc = tf.keras.constraints


class SparseAmsoftmaxLoss(kls.Loss):

  def __init__(self,
               scale: int = 30,
               margin: int = 0.35,
               batch_size: int = None,
               reduction='auto',
               name=None):
    """ sparse addivate margin softmax

        Parameters
        ----------

        scale : int, optional

            by default 30

        margin : int, optional

            by default 0.35

    """
    super().__init__(reduction=reduction, name=name)
    self.scale = scale
    self.margin = margin
    if batch_size:
      self.batch_idxs = tf.expand_dims(
          tf.range(0, batch_size, dtype=tf.int32), 1)  # shape [batch,1]

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    idxs = tf.concat([self.batch_idxs, tf.cast(y_true, tf.int32)], 1)
    y_true_pred = tf.gather_nd(y_pred, idxs)
    y_true_pred = tf.expand_dims(y_true_pred, 1)
    y_true_pred_margin = y_true_pred - self.margin
    _Z = tf.concat([y_pred, y_true_pred_margin], 1)
    _Z = _Z * self.scale
    logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims=True)
    logZ = logZ + tf.math.log(1 - tf.math.exp(self.scale * y_true_pred - logZ))
    return -y_true_pred_margin * self.scale + logZ


class AmsoftmaxLoss(SparseAmsoftmaxLoss):

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = (y_true * (y_pred - self.margin) + (1-y_true) * y_pred) * self.scale
    return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)


class CircleLoss(kls.Loss):

  def __init__(self,
               gamma: int = 64,
               margin: float = 0.25,
               batch_size: int = None,
               reduction='auto',
               name=None):
    super().__init__(reduction=reduction, name=name)
    self.gamma = gamma
    self.margin = margin
    self.O_p = 1 + self.margin
    self.O_n = -self.margin
    self.Delta_p = 1 - self.margin
    self.Delta_n = self.margin
    if batch_size:
      self.batch_size = batch_size
      self.batch_idxs = tf.expand_dims(
          tf.range(0, batch_size, dtype=tf.int32), 1)  # shape [batch,1]

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ NOTE : y_pred must be cos similarity
    
    Args:
        y_true (tf.Tensor): shape [batch,ndim]
        y_pred (tf.Tensor): shape [batch,ndim]
    
    Returns:
        tf.Tensor: loss
    """
    alpha_p = tf.nn.relu(self.O_p - tf.stop_gradient(y_pred))
    alpha_n = tf.nn.relu(tf.stop_gradient(y_pred) - self.O_n)
    # yapf: disable
    y_pred = (y_true * (alpha_p * (y_pred - self.Delta_p)) +
          (1-y_true) * (alpha_n * (y_pred - self.Delta_n))) * self.gamma
    # yapf: enable
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


class SparseCircleLoss(CircleLoss):

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ NOTE : y_pred must be cos similarity
    
    Args:
        y_true (tf.Tensor): shape [batch,ndim]
        y_pred (tf.Tensor): shape [batch,ndim]
    
    Returns:
        tf.Tensor: loss
    """

    # idxs = tf.concat([self.batch_idxs, tf.cast(y_true, tf.int32)], 1)
    # sp = tf.expand_dims(tf.gather_nd(y_pred, idxs), 1)

    # alpha_p = tf.nn.relu(self.O_p - tf.stop_gradient(sp))
    # alpha_n = tf.nn.relu(tf.stop_gradient(y_pred) - self.O_n)
    # alpha_n_for_p = tf.expand_dims(tf.gather_nd(alpha_n, idxs), 1)

    # r_sp_m = alpha_p * (sp - self.Delta_p)
    # r_sn_m = alpha_n * (y_pred - self.Delta_n)
    # _Z = tf.concat([r_sn_m, r_sp_m], 1)
    # _Z = _Z * self.gamma
    # # sum all similarity
    # logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims=True)
    # # remove sn_p from all sum similarity
    # TODO This line will be numerical overflow, Need a more numerically safe method
    # logZ = logZ + tf.math.log(1 - tf.math.exp(
    #     (alpha_n_for_p * (sp - self.Delta_n)) * self.gamma - logZ))

    # return -r_sp_m * self.gamma + logZ
    idxs = tf.concat([self.batch_idxs, tf.cast(y_true, tf.int32)], 1)
    sp = tf.expand_dims(tf.gather_nd(y_pred, idxs), 1)
    mask = tf.logical_not(
        tf.scatter_nd(idxs, tf.ones(tf.shape(idxs)[0], tf.bool),
                      tf.shape(y_pred)))

    sn = tf.reshape(tf.boolean_mask(y_pred, mask), (self.batch_size, -1))

    alpha_p = tf.nn.relu(self.O_p - tf.stop_gradient(sp))
    alpha_n = tf.nn.relu(tf.stop_gradient(sn) - self.O_n)

    r_sp_m = alpha_p * (sp - self.Delta_p)
    r_sn_m = alpha_n * (sn - self.Delta_n)
    _Z = tf.concat([r_sn_m, r_sp_m], 1)
    _Z = _Z * self.gamma
    # sum all similarity
    logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims=True)
    # remove sn_p from all sum similarity
    return -r_sp_m * self.gamma + logZ


class PairCircleLoss(CircleLoss):

  def call(self, sp: tf.Tensor, sn: tf.Tensor) -> tf.Tensor:
    """ use within-class similarity and between-class similarity for loss
    
    Args:
        sp (tf.Tensor): within-class similarity  shape [batch, K]
        sn (tf.Tensor): between-class similarity shape [batch, L]
    
    Returns:
        tf.Tensor: loss
    """
    ap = tf.nn.relu(-tf.stop_gradient(sp) + 1 + self.margin)
    an = tf.nn.relu(tf.stop_gradient(sn) + self.margin)

    logit_p = -ap * (sp - self.Delta_p) * self.gamma
    logit_n = an * (sn - self.Delta_n) * self.gamma

    return tf.math.softplus(
        tf.math.reduce_logsumexp(logit_n, axis=-1, keepdims=True) +
        tf.math.reduce_logsumexp(logit_p, axis=-1, keepdims=True))


if __name__ == "__main__":
  batch_size = 8
  nclass = 10
  y_true = tf.random.uniform((batch_size,), 0, nclass, dtype=tf.int32)
  y_pred = tf.random.uniform((batch_size, nclass), -1, 1, dtype=tf.float32)
  batch_idxs = tf.expand_dims(tf.range(0, batch_size, dtype=tf.int32),
                              1)  # shape [batch,1]
  idxs = tf.concat([batch_idxs, tf.cast(tf.expand_dims(y_true, -1), tf.int32)], 1)
  mask = tf.logical_not(
      tf.scatter_nd(idxs, tf.ones(tf.shape(idxs)[0], tf.bool), tf.shape(y_pred)))

  sp = tf.expand_dims(tf.gather_nd(y_pred, idxs), 1)
  sn = tf.reshape(tf.boolean_mask(y_pred, mask), (batch_size, -1))

  circleloss = CircleLoss()
  sparsecircleloss = SparseCircleLoss(batch_size=batch_size)
  paircircleloss = PairCircleLoss()

  print(
      'circle loss:\n',
      circleloss.call(tf.one_hot(y_true, nclass, dtype=tf.float32),
                      y_pred).numpy())
  print('sparse circle loss:\n',
        sparsecircleloss.call(tf.expand_dims(y_true, -1), y_pred).numpy().ravel())
  print('pair circle loss:\n', paircircleloss.call(sp, sn).numpy().ravel())
