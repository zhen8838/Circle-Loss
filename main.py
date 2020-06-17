import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
from circle_loss import SparseAmsoftmaxLoss, SparseCircleLoss, CircleLoss, PairCircleLoss, ProxyAnchorLoss
import matplotlib.pyplot as plt
import cycler
import numpy as np

plt.style.use('seaborn-paper')
plt.rcParams['axes.prop_cycle'] = cycler.cycler(
    'color', plt.cm.tab10(np.linspace(0, 1, 9)))
plt.rc('font', **{'weight': 'bold', 'size': '13'})
plt.rc('axes', unicode_minus=False)
params = {
    'legend.fontsize': 'small',
    'axes.labelsize': 'small',
    'axes.titlesize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small'
}
plt.rcParams.update(params)

if __name__ == "__main__":

  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  batch_size = 500  # Note Must be divisible by 50000
  tf.random.set_seed(10101)
  np.random.seed(10101)
  (train_x, train_y), (test_x, test_y) = k.datasets.cifar10.load_data()
  train_x = ((train_x - 127.5) / 127.5).astype('float32')
  test_x = ((test_x - 127.5) / 127.5).astype('float32')

  softmax_model: k.Model = k.Sequential([
      kl.Input(shape=(32, 32, 3)),
      kl.Conv2D(64, kernel_size=(3, 3), padding='SAME'),
      kl.BatchNormalization(),
      kl.ReLU(6),
      kl.MaxPooling2D((2, 2)),
      kl.Conv2D(128, kernel_size=(3, 3), padding='SAME'),
      kl.BatchNormalization(),
      kl.ReLU(6),
      kl.MaxPooling2D((2, 2)),
      kl.Conv2D(256, kernel_size=(3, 3), padding='SAME'),
      kl.BatchNormalization(),
      kl.ReLU(6),
      kl.MaxPooling2D((2, 2)),
      kl.Conv2D(256, kernel_size=(3, 3), padding='SAME'),
      kl.BatchNormalization(),
      kl.ReLU(6),
      kl.Conv2D(128, kernel_size=(3, 3), padding='SAME'),
      kl.BatchNormalization(),
      kl.ReLU(6),
      kl.GlobalMaxPooling2D(),
      kl.Dense(128),
      # kl.BatchNormalization(),
      # kl.ReLU(6),
      # kl.Dense(3),
      kl.Lambda(lambda x: tf.nn.l2_normalize(x, 1), name='emmbeding'),
      kl.Dense(10, use_bias=False, kernel_constraint=k.constraints.unit_norm())
  ])
  ams_model = k.models.clone_model(softmax_model)
  circle_model = k.models.clone_model(softmax_model)
  proxy_model = k.models.clone_model(softmax_model)

  softmax_model.compile(
      loss=k.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=k.optimizers.Adam(),
      metrics=[k.metrics.SparseCategoricalAccuracy('acc')])

  ams_model.compile(
      loss=SparseAmsoftmaxLoss(batch_size=batch_size),
      optimizer=k.optimizers.Adam(),
      metrics=[k.metrics.SparseCategoricalAccuracy('acc')])

  circle_model.compile(
      loss=SparseCircleLoss(batch_size=batch_size),
      optimizer=k.optimizers.Adam(),
      metrics=[k.metrics.SparseCategoricalAccuracy('acc')])

  proxy_model.compile(
      loss=ProxyAnchorLoss(batch_size=batch_size),
      optimizer=k.optimizers.Adam(),
      metrics=[k.metrics.CategoricalAccuracy('acc')])

  if not tf.io.gfile.exists('softmax_loss.h5'):
    softmax_history = softmax_model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=20,
        validation_data=(test_x, test_y))
    softmax_model.save('softmax_loss.h5')
    plt.plot(
        softmax_history.epoch,
        softmax_history.history['val_acc'],
        label="softmax")
  else:
    softmax_model.load_weights('softmax_loss.h5')

  if not tf.io.gfile.exists('ams_loss.h5'):
    ams_history = ams_model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=20,
        validation_data=(test_x, test_y))
    ams_model.save('ams_loss.h5')
    plt.plot(
        ams_history.epoch, ams_history.history['val_acc'], label="am-softmax")
  else:
    ams_model.load_weights('ams_loss.h5')

  if not tf.io.gfile.exists('circle_loss.h5'):

    circle_history = circle_model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=20,
        validation_data=(test_x, test_y))
    circle_model.save('circle_loss.h5')
    plt.plot(
        circle_history.epoch,
        circle_history.history['val_acc'],
        label="circle loss")
  else:
    circle_model.load_weights('circle_loss.h5')

  if not tf.io.gfile.exists('proxy_loss.h5'):
    proxy_history = proxy_model.fit(
        x=train_x,
        y=tf.keras.utils.to_categorical(train_y, 10),
        batch_size=batch_size,
        epochs=20,
        validation_data=(test_x, tf.keras.utils.to_categorical(test_y, 10)))
    proxy_model.save('proxy_loss.h5')
    plt.plot(
        proxy_history.epoch,
        proxy_history.history['val_acc'],
        label="proxy loss")
    plt.legend(loc='upper left')
    plt.title('Validation Accuracy')
    plt.tight_layout()
    plt.savefig(
        'benchmark.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()
  else:
    proxy_model.load_weights('proxy_loss.h5')

  print('Softmax evaluate:')
  softmax_model.evaluate(test_x, test_y, batch_size=batch_size)
  print('Am Softmax evaluate:')
  ams_model.evaluate(test_x, test_y, batch_size=batch_size)
  print('Circle Loss evaluate:')
  circle_model.evaluate(test_x, test_y, batch_size=batch_size)
  print('Proxy Loss evaluate:')
  proxy_model.evaluate(test_x, tf.keras.utils.to_categorical(test_y, 10), batch_size=batch_size)
