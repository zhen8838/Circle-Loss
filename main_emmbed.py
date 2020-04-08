import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
from circle_loss import SparseAmsoftmaxLoss, SparseCircleLoss
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cycler
import numpy as np
from typing import List

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


def build_ball(ax):
  xlm = ax.get_xlim3d()
  ylm = ax.get_ylim3d()
  zlm = ax.get_zlim3d()
  ax.set_xlim3d(-.82, 0.82)
  ax.set_ylim3d(-.82, 0.82)
  ax.set_zlim3d(-.82, 0.82)
  # First remove fill
  ax.xaxis.pane.fill = False
  ax.yaxis.pane.fill = False
  ax.zaxis.pane.fill = False

  # Now set color to white (or whatever is "invisible")
  ax.xaxis.pane.set_edgecolor('w')
  ax.yaxis.pane.set_edgecolor('w')
  ax.zaxis.pane.set_edgecolor('w')

  # Bonus: To get rid of the grid as well:
  ax.grid(False)

  ax.set_xticks([-0.5, 0, 0.5])
  ax.set_yticks([-0.5, 0, 0.5])
  ax.set_zticks([-1, -0.5, 0, 0.5, 1])

  u = np.linspace(0, 2 * np.pi, 15)
  v = np.linspace(0, np.pi, 20)
  x = 1 * np.outer(np.cos(u), np.sin(v))
  y = 1 * np.outer(np.sin(u), np.sin(v))
  z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
  ax.plot_wireframe(
      x, y, z, colors='dimgray', alpha=0.6, linestyles='-', linewidths=1)


if __name__ == "__main__":

  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  batch_size = 500  # Note Must be divisible by 50000
  
  tf.random.set_seed(10101)
  np.random.seed(10101)
  
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
      kl.BatchNormalization(),
      kl.ReLU(6),
      kl.Dense(3),
      kl.Lambda(lambda x: tf.nn.l2_normalize(x, 1), name='emmbeding'),
      kl.Dense(10, use_bias=False, kernel_constraint=k.constraints.unit_norm())
  ])
  circle_model = k.models.clone_model(ams_model)

  ams_model.compile(
      loss=SparseAmsoftmaxLoss(batch_size=batch_size),
      optimizer=k.optimizers.Adam(),
      metrics=[k.metrics.SparseCategoricalAccuracy('acc')])
  if not tf.io.gfile.exists('ams_loss_emmbed.h5'):
    ams_model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=20,
        validation_data=(test_x, test_y))
    ams_model.save('ams_loss_emmbed.h5')
  else:
    ams_model.load_weights('ams_loss_emmbed.h5')

  circle_model.compile(
      loss=SparseCircleLoss(batch_size=batch_size),
      optimizer=k.optimizers.Adam(),
      metrics=[k.metrics.SparseCategoricalAccuracy('acc')])

  if not tf.io.gfile.exists('circle_loss_emmbed.h5'):

    circle_model.fit(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        epochs=20,
        validation_data=(test_x, test_y))
    circle_model.save('circle_loss_emmbed.h5')
  else:
    circle_model.load_weights('circle_loss_emmbed.h5')
  print('Am Softmax evaluate:')
  ams_model.evaluate(test_x, test_y, batch_size=batch_size)
  print('Circle Loss evaluate:')
  circle_model.evaluate(test_x, test_y, batch_size=batch_size)

  fig = plt.figure(figsize=[1.3 * i for i in [8, 4]])
  ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # type: Axes3D
  ax2 = fig.add_subplot(1, 2, 2, projection='3d')  # type: Axes3D
  ax1.view_init(elev=25., azim=120.)
  ax2.view_init(elev=25., azim=120.)
  build_ball(ax1)
  build_ball(ax2)

  axs = [ax1, ax2]
  titles = ['Am Softmax', 'Circle Loss']
  models: List[k.Model] = [ams_model, circle_model]
  num = 500
  n = 10

  for i in range(2):
    # 加载数据
    model = models[i]
    encoder = k.backend.function(model.inputs[0],
                                 model.get_layer('emmbeding').output)
    with k.backend.learning_phase_scope(0):
      vec = []
      tures = []
      preds = []
      for j in range(len(test_y) // batch_size):
        vec.append(encoder(test_x[j * batch_size:(j+1) * batch_size]))
        tures.append(test_y[j * batch_size:(j+1) * batch_size])
        preds.append(
            model(test_x[j * batch_size:(j+1) * batch_size], training=False))
    vec = np.concatenate(vec)
    tures = np.concatenate(tures).ravel().astype('int32')
    preds = tf.argmax(tf.nn.softmax(np.concatenate(preds), -1), -1).numpy()

    # 设置颜色循环
    # NOTE 找到所有预测正确，且y_true等于指定数
    res = []
    for j in range(n):
      # boolmask = np.logical_and((preds == tures), tures == j)
      boolmask = (preds == j)
      valid = np.where(boolmask == True)[0]
      if len(valid) > 0:
        idx = np.random.choice(valid, num)
        res.append(vec[idx])
      else:
        res.append([])

    for j in range(n):
      if len(res[j]) > 0:
        axs[i].scatter(res[j][:, 0], res[j][:, 1], res[j][:, 2], label=f'{j}')
    axs[i].set_title(titles[i])

  plt.tight_layout()
  fig.savefig(
      'emmbeding.png', transparent=True, bbox_inches='tight', pad_inches=0)
  plt.show()
