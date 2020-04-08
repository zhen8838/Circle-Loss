import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
from circle_loss import SparseAmsoftmaxLoss, SparseCircleLoss, CircleLoss, PairCircleLoss

if __name__ == "__main__":

  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  batch_size = 500  # Note Must be divisible by 50000
  tf.random.set_seed(10101)

  (train_x, train_y), (test_x, test_y) = k.datasets.cifar10.load_data()
  train_x = ((train_x-127.5) / 127.5).astype('float32')
  test_x = ((test_x-127.5) / 127.5).astype('float32')

  ams_model: k.Model = k.Sequential([
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
  circle_model = k.models.clone_model(ams_model)

  ams_model.compile(
      loss=SparseAmsoftmaxLoss(batch_size=batch_size),
      optimizer=k.optimizers.Adam(),
      metrics=[k.metrics.SparseCategoricalAccuracy('acc')])
  if not tf.io.gfile.exists('ams_loss.h5'):
    ams_model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=20,
        validation_data=(test_x, test_y))
    ams_model.save('ams_loss.h5')
  else:
    ams_model.load_weights('ams_loss.h5')

  circle_model.compile(
      loss=SparseCircleLoss(batch_size=batch_size),
      optimizer=k.optimizers.Adam(),
      metrics=[k.metrics.SparseCategoricalAccuracy('acc')])

  if not tf.io.gfile.exists('circle_loss.h5'):

    circle_model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=20,
        validation_data=(test_x, test_y))
    circle_model.save('circle_loss.h5')
  else:
    circle_model.load_weights('circle_loss.h5')
  print('Am Softmax evaluate:')
  ams_model.evaluate(test_x, test_y, batch_size=batch_size)
  print('Circle Loss evaluate:')
  circle_model.evaluate(test_x, test_y, batch_size=batch_size)
