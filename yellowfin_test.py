import os
import mxnet as mx
import numpy as np
from yellowfin import *
import time


n_dim = 1000000
n_iter = 50

class MyLoss(mx.operator.NumpyOp):
  def __init__(self):
    super(MyLoss, self).__init__(False)

  def list_arguments(self):
    return ['data', 'label']

  def list_outputs(self):
    return ['output']

  def infer_shape(self, in_shape):
    data_shape = in_shape[0]
    label_shape = (in_shape[0][0],)
    output_shape = in_shape[0]
    return [data_shape, label_shape], [output_shape]

  def forward(self, in_data, out_data):
    x = in_data[0]
    y = out_data[0]
    y[:] = x

  def backward(self, out_grad, in_data, out_data, in_grad):
    dx = in_grad[0]
    dx[:] = np.ones(dx.shape)

@mx.initializer.register
class CustomInit(mx.initializer.Initializer):
    """Initializes the weights to a scalar value.
    Parameters
    ----------
    value : float
        Fill value.
    """
    def __init__(self):
        super(CustomInit, self).__init__()

    def _init_weight(self, _, arr):
        arr[:] = 1

    def _init_bias(self, _, arr):
        arr[:] = 1

def tune_everything(x0squared, C, T, gmin, gmax):
  # First tune based on dynamic range
  if C == 0:
    dr = gmax / gmin
    mustar = ((np.sqrt(dr) - 1) / (np.sqrt(dr) + 1)) ** 2
    alpha_star = (1 + np.sqrt(mustar)) ** 2 / gmax

    return alpha_star, mustar

  dist_to_opt = x0squared
  grad_var = C
  max_curv = gmax
  min_curv = gmin
  const_fact = dist_to_opt * min_curv ** 2 / 2 / grad_var
  coef = [-1, 3, -(3 + const_fact), 1]
  roots = np.roots(coef)
  roots = roots[np.real(roots) > 0]
  roots = roots[np.real(roots) < 1]
  root = roots[np.argmin(np.imag(roots))]

  assert root > 0 and root < 1 and np.absolute(root.imag) < 1e-6

  dr = max_curv / min_curv
  assert max_curv >= min_curv
  mu = max(((np.sqrt(dr) - 1) / (np.sqrt(dr) + 1)) ** 2, root ** 2)

  lr_min = (1 - np.sqrt(mu)) ** 2 / min_curv
  lr_max = (1 + np.sqrt(mu)) ** 2 / max_curv

  alpha_star = lr_min
  mustar = mu

  return alpha_star, mustar

def test_measurement(zero_debias=True):

  data = np.array([np.ones(n_dim)])
  label = np.array([0])
  batch_size = 1
  train_iter = mx.io.NDArrayIter(data, label, batch_size, label_name='linear_output_label')

  net = mx.sym.Variable('data')
  weight = mx.sym.Variable(name='fc1_weight')
  bias = mx.sym.Variable(name='fc1_bias')
  net = mx.sym.FullyConnected(data=net, weight=weight, bias=bias, name='fc1', num_hidden=1)
  myloss = MyLoss()
  net = myloss(data=net, name='linear_output')

  mod = mx.mod.Module(symbol=net,
                      context=mx.cpu(),
                      data_names=['data'],
                      label_names=['linear_output_label'])
  # allocate memory given the input data and label shapes
  mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
  # initialize parameters by uniform random numbers
  mod.init_params(CustomInit())
  # use SGD with learning rate 0.1 to train
  mod.init_optimizer(optimizer='YFOptimizer', optimizer_params=(('learning_rate', 1.0), ('momentum', 0.0), ('zero_bias', zero_debias)))
  # use accuracy as the metric
  metric = mx.metric.create('mse')

  target_h_max = 0.0
  target_h_min = 0.0
  g_norm_squared_avg = 0.0
  g_norm_avg = 0.0
  g_avg = 0.0
  target_dist = 0.0

  # train 5 epochs, i.e. going over the data iter one pass
  for epoch in range(n_iter):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
      i = epoch
      mod.forward(batch, is_train=True)  # compute predictions
      mod.update_metric(metric, batch.label)  # accumulate prediction MSE
      mod.backward()  # compute gradients
      mod._exec_group.grad_arrays[0][0] *= i + 1
      mod._exec_group.grad_arrays[1][0] *= i + 1
      mod.update()  # update parameters

      res = mod._optimizer._test_res

      g_norm_squared_avg = 0.999 * g_norm_squared_avg \
                           + 0.001 * np.sum(((i + 1) * np.ones([n_dim + 1, ])) ** 2)
      g_norm_avg = 0.999 * g_norm_avg \
                   + 0.001 * np.linalg.norm((i + 1) * np.ones([n_dim + 1, ]))
      g_avg = 0.999 * g_avg + 0.001 * (i + 1)

      target_h_max = 0.999 * target_h_max + 0.001 * (i + 1) ** 2 * (n_dim + 1)
      target_h_min = 0.999 * target_h_min + 0.001 * max(1, i + 2 - 20) ** 2 * (n_dim + 1)
      if zero_debias:
        target_var = g_norm_squared_avg / (1 - 0.999 ** (i + 1)) \
                     - g_avg ** 2 * (n_dim + 1) / (1 - 0.999 ** (i + 1)) ** 2
      else:
        target_var = g_norm_squared_avg - g_avg ** 2 * (n_dim + 1)
      target_dist = 0.999 * target_dist + 0.001 * g_norm_avg / g_norm_squared_avg

      if i == 0:
        continue
      if zero_debias:
        # print "iter ", i, " h max ", res[0], target_h_max/(1-0.999**(i + 1) ), \
        #   " h min ", res[1], target_h_min/(1-0.999**(i + 1) ), \
        #   " var ", res[2], target_var, \
        #   " dist ", res[3], target_dist/(1-0.999**(i + 1) )
        assert np.abs(target_h_max / (1 - 0.999 ** (i + 1)) - res[0]) < np.abs(res[0]) * 1e-3
        assert np.abs(target_h_min / (1 - 0.999 ** (i + 1)) - res[1]) < np.abs(res[1]) * 1e-3
        assert np.abs(target_var - res[2]) < np.abs(target_var) * 1e-3
        assert np.abs(target_dist / (1 - 0.999 ** (i + 1)) - res[3]) < np.abs(res[3]) * 1e-3
      else:
        # print "iter ", i, " h max ", res[0], target_h_max, " h min ", res[1], target_h_min, \
        #   " var ", res[2], target_var, " dist ", res[3], target_dist
        assert np.abs(target_h_max - res[0]) < np.abs(target_h_max) * 1e-3
        assert np.abs(target_h_min - res[1]) < np.abs(target_h_min) * 1e-3
        assert np.abs(target_var - res[2]) < np.abs(res[2]) * 1e-3
        assert np.abs(target_dist - res[3]) < np.abs(res[3]) * 1e-3

  print "sync measurement test passed!"


def test_lr_mu(zero_debias=False):

  data = np.array([np.ones(n_dim)])
  label = np.array([0])
  batch_size = 1
  train_iter = mx.io.NDArrayIter(data, label, batch_size, label_name='linear_output_label')

  net = mx.sym.Variable('data')
  weight = mx.sym.Variable(name='fc1_weight')
  bias = mx.sym.Variable(name='fc1_bias')
  net = mx.sym.FullyConnected(data=net, weight=weight, bias=bias, name='fc1', num_hidden=1)
  myloss = MyLoss()
  net = myloss(data=net, name='linear_output')

  mod = mx.mod.Module(symbol=net,
                      context=mx.cpu(),
                      data_names=['data'],
                      label_names=['linear_output_label'])
  # allocate memory given the input data and label shapes
  mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
  # initialize parameters by uniform random numbers
  mod.init_params(CustomInit())
  # use SGD with learning rate 0.1 to train
  mod.init_optimizer(optimizer='YFOptimizer', optimizer_params=(('learning_rate', 1.0), ('momentum', 0.0), ('zero_bias', zero_debias)))
  # use accuracy as the metric
  metric = mx.metric.create('mse')

  target_h_max = 0.0
  target_h_min = 0.0
  g_norm_squared_avg = 0.0
  g_norm_avg = 0.0
  g_avg = 0.0
  target_dist = 0.0
  target_lr = 1.0
  target_mu = 0.0

  # train 5 epochs, i.e. going over the data iter one pass
  for epoch in range(n_iter):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
      i = epoch

      mod.forward(batch, is_train=True)  # compute predictions
      mod.update_metric(metric, batch.label)  # accumulate prediction MSE
      mod.backward()  # compute gradients
      mod._exec_group.grad_arrays[0][0] *= i + 1
      mod._exec_group.grad_arrays[1][0] *= i + 1
      mod.update()  # update parameters

      res = mod._optimizer._test_res

      g_norm_squared_avg = 0.999 * g_norm_squared_avg \
                           + 0.001 * np.sum(((i + 1) * np.ones([n_dim + 1, ])) ** 2)
      g_norm_avg = 0.999 * g_norm_avg \
                   + 0.001 * np.linalg.norm((i + 1) * np.ones([n_dim + 1, ]))
      g_avg = 0.999 * g_avg + 0.001 * (i + 1)

      target_h_max = 0.999 * target_h_max + 0.001 * (i + 1) ** 2 * (n_dim + 1)
      target_h_min = 0.999 * target_h_min + 0.001 * max(1, i + 2 - 20) ** 2 * (n_dim + 1)
      if zero_debias:
        target_var = g_norm_squared_avg / (1 - 0.999 ** (i + 1)) \
                     - g_avg ** 2 * (n_dim + 1) / (1 - 0.999 ** (i + 1)) ** 2
      else:
        target_var = g_norm_squared_avg - g_avg ** 2 * (n_dim + 1)
      target_dist = 0.999 * target_dist + 0.001 * g_norm_avg / g_norm_squared_avg

      if i == 0:
        continue
      if zero_debias:
        # print "iter ", i, " h max ", res[0], target_h_max/(1-0.999**(i + 1) ), \
        #   " h min ", res[1], target_h_min/(1-0.999**(i + 1) ), \
        #   " var ", res[2], target_var, \
        #   " dist ", res[3], target_dist/(1-0.999**(i + 1) )
        assert np.abs(target_h_max / (1 - 0.999 ** (i + 1)) - res[0]) < np.abs(res[0]) * 1e-3
        assert np.abs(target_h_min / (1 - 0.999 ** (i + 1)) - res[1]) < np.abs(res[1]) * 1e-3
        assert np.abs(target_var - res[2]) < np.abs(target_var) * 1e-3
        assert np.abs(target_dist / (1 - 0.999 ** (i + 1)) - res[3]) < np.abs(res[3]) * 1e-3
      else:
        # print "iter ", i, " h max ", res[0], target_h_max, " h min ", res[1], target_h_min, \
        # " var ", res[2], target_var, " dist ", res[3], target_dist
        assert np.abs(target_h_max - res[0]) < np.abs(target_h_max) * 1e-3
        assert np.abs(target_h_min - res[1]) < np.abs(target_h_min) * 1e-3
        assert np.abs(target_var - res[2]) < np.abs(res[2]) * 1e-3
        assert np.abs(target_dist - res[3]) < np.abs(res[3]) * 1e-3

      if i > 0:
        if zero_debias:
          lr, mu = tune_everything((target_dist / (1 - 0.999 ** (i + 1))) ** 2,
                                   target_var, 1, target_h_min / (1 - 0.999 ** (i + 1)),
                                   target_h_max / (1 - 0.999 ** (i + 1)))
        else:
          lr, mu = tune_everything(target_dist ** 2, target_var, 1, target_h_min, target_h_max)
        lr = np.real(lr)
        mu = np.real(mu)
        target_lr = 0.999 * target_lr + 0.001 * lr
        target_mu = 0.999 * target_mu + 0.001 * mu
        # print "lr ", target_lr, res[4], " mu ", target_mu, res[5]
        assert target_lr == 0.0 or np.abs(target_lr - res[4]) < np.abs(res[4]) * 1e-3
        assert target_mu == 0.0 or np.abs(target_mu - res[5]) < np.abs(res[5]) * 5e-3
  print "lr and mu computing test passed!"


if __name__ == "__main__":
  start = time.time()
  test_measurement(zero_debias=False)
  end = time.time()
  print "measurement test without zero_debias done in ", (end - start) / float(n_iter), " s/iter!"

  start = time.time()
  test_measurement(zero_debias=True)
  end = time.time()
  print "measurement test with zero_debias done in ", (end - start) / float(n_iter), " s/iter!"

  start = time.time()
  test_lr_mu(zero_debias=False)
  end = time.time()
  print "lr and mu test done with zero_debias in ", (end - start) / float(n_iter), " s/iter!"

  start = time.time()
  test_lr_mu(zero_debias=True)
  end = time.time()
  print "lr and mu test done with zero_debias in ", (end - start) / float(n_iter), " s/iter!"
