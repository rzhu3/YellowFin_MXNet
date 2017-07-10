import numpy as np
import math
import mxnet as mx
from mxnet.ndarray import zeros, NDArray, square, sqrt

@mx.optimizer.Optimizer.register
class YFOptimizer(mx.optimizer.Optimizer):
  """The YF optimizer built upon SGD optimizer with momentum and weight decay.
  The optimizer updates the weight by::
    state = momentum * state + lr * rescale_grad * clip(grad, clip_gradient) + wd * weight
    weight = weight - state
  For details of the update algorithm see :class:`~mxnet.ndarray.sgd_update` and
  :class:`~mxnet.ndarray.sgd_mom_update`.
  This optimizer accepts the following parameters in addition to those accepted
  by :class:`.Optimizer`.
  Parameters
  ----------
  momentum : float, optional
    The initial momentum value.
  beta : float, optional
    The smoothing parameter for estimations.
  curv_win_width: int, optional

  zero_bias: bool, optional
  """

  def __init__(self, momentum=0.0, beta=0.999, curv_win_width=10, zero_bias=True, **kwargs):
    super(YFOptimizer, self).__init__(**kwargs)
    self.momentum = momentum
    self.mom_dict = {}
    self.lr_dict = {}
    self.beta = beta
    self.curv_win_width = curv_win_width
    self.zero_bias = zero_bias
    # The following are global states for YF tuner
    # self._iter = 0
    self._global_state = {}
    self._global_state['h_min_avg'] = {}
    self._global_state['h_max_avg'] = {}
    self._global_state['curv_win'] = {}
    self._global_state['grad_norm_avg'] = {}
    self._global_state['grad_norm_squared'] = {}
    self._global_state['grad_norm_squared_avg'] = {}
    self._global_state['dist_to_opt_avg'] = {}
    self._h_min = 0.0
    self._h_max = 0.0
    self._lr = self.lr
    self._mom = momentum

  def _get_mom(self, index):
    if index in self.mom_dict:
      return self.mom_dict[index]
    else:
      return 0.0

  def _get_lr_dict(self, index):
    lr = self.lr
    if index in self.lr_dict:
      return self.lr_dict[index]
    else:
      return lr

  def create_state(self, index, weight):
    # 2. These two state tensors are used in grad variance estimation
    # grad_avg = None
    # grad_avg_squared = None
    momentum = zeros(weight.shape, weight.context, dtype=weight.dtype)
    grad_avg = zeros(weight.shape, weight.context, dtype=weight.dtype)
    # return momentum, grad_avg
    grad_avg_squared = zeros(weight.shape, weight.context, dtype=weight.dtype)
    return momentum, grad_avg, grad_avg_squared

  def zero_debias_factor(self, t):
    if not self.zero_bias:
      return 1.0
    return 1.0 - self.beta ** (t + 1)

  def curvature_range(self, index, t):
    global_state = self._global_state
    if t == 0:
      global_state['curv_win'][index] = mx.ndarray.zeros(self.curv_win_width)
    curv_win = global_state['curv_win'][index]
    grad_norm_squared = self._global_state['grad_norm_squared'][index]
    curv_win[t % self.curv_win_width] = grad_norm_squared
    valid_end = min(self.curv_win_width, t + 1)
    beta = self.beta
    if t == 0:
      global_state['h_min_avg'][index] = 0.0
      global_state['h_max_avg'][index] = 0.0
    global_state['h_min_avg'][index] = \
      beta*global_state['h_min_avg'][index] + (1-beta)*mx.ndarray.min(curv_win[:valid_end])
    global_state['h_max_avg'][index] = \
      beta*global_state['h_max_avg'][index] + (1-beta)*mx.ndarray.max(curv_win[:valid_end])
    if self.zero_bias:
      debias_factor = self.zero_debias_factor(t)
      h_min = global_state['h_min_avg'][index].asscalar() / debias_factor
      h_max = global_state['h_max_avg'][index].asscalar() / debias_factor
    else:
      h_min = global_state['h_min_avg'][index].asscalar()
      h_max = global_state['h_max_avg'][index].asscalar()
    return h_min, h_max

  def grad_variance(self, index, grad, state, t):
    beta = self.beta
    global_state = self._global_state

    _, grad_avg, grad_avg_squared = state
    # _, grad_avg = state
    grad_avg[:] = beta * grad_avg + (1 - beta) * grad
    # grad_avg_squared[:] = beta * grad_avg_squared + (1 - beta) * grad * grad
    self._grad_var = mx.ndarray.sum(grad_avg * grad_avg)

    if self.zero_bias:
      debias_factor = self.zero_debias_factor(t)
    else:
      debias_factor = 1.0
    self._grad_var /= -(debias_factor ** 2)
    # self._grad_var += mx.ndarray.sum(grad_avg) ** 2 / debias_factor
    self._grad_var += global_state['grad_norm_squared_avg'][index] / debias_factor
    # self._grad_var = self._grad_var.asscalar()
    return mx.ndarray.abs(self._grad_var).asscalar()

  def dist_to_opt(self, index, t):
    beta = self.beta
    global_state = self._global_state
    if t == 0:
      global_state['grad_norm_avg'][index] = 0.0
      global_state["dist_to_opt_avg"][index] = 0.0
    global_state["grad_norm_avg"][index] = \
      global_state["grad_norm_avg"][index] * beta + (1 - beta) * math.sqrt(global_state["grad_norm_squared"][index] )
    global_state["dist_to_opt_avg"][index] = \
      global_state["dist_to_opt_avg"][index] * beta \
      + (1 - beta) * global_state["grad_norm_avg"][index] / global_state['grad_norm_squared_avg'][index]
    if self.zero_bias:
      debias_factor = self.zero_debias_factor(t)
      return global_state["dist_to_opt_avg"][index] / debias_factor
    else:
      return global_state["dist_to_opt_avg"][index]

  def single_step_mu_lr(self, C, D, h_min, h_max):
    coef = [-1.0, 3.0, 0.0, 1.0]
    coef[2] = -(3 + D**2 * h_min**2 / 2 / C)
    roots = np.roots(coef)
    root = roots[np.logical_and(np.logical_and(np.real(roots) > 0.0,
      np.real(roots) < 1.0), np.imag(roots) < 1e-5) ]
    if root.size != 1:
      # print self._grad_var, self._dist_to_opt, self._h_min, self._h_max
      raise ValueError
    dr = h_max / h_min
    mu_t = max(np.real(root)[0]**2, ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2 )
    lr_t = (1.0 - math.sqrt(mu_t)) ** 2 / h_min
    return mu_t, lr_t

  def after_apply(self, index, grad, state, t, lr, momentum):
    beta = self.beta
    global_state = self._global_state
    if t == 0:
      global_state['grad_norm_squared_avg'][index] = 0.0

    global_state['grad_norm_squared'][index] = mx.ndarray.sum(grad * grad).asscalar()
    global_state['grad_norm_squared_avg'][index] = \
      global_state['grad_norm_squared_avg'][index] * beta + (1 - beta) * global_state['grad_norm_squared'][index]

    h_min, h_max = self.curvature_range(index, t)
    C = self.grad_variance(index, grad, state, t)
    D = self.dist_to_opt(index, t)
    if t > 0:
      mu_t, lr_t = self.single_step_mu_lr(C, D, h_min, h_max)
      self.mom_dict[index] = beta * momentum + (1 - beta) * mu_t
      self.lr_dict[index] = beta * lr + (1 - beta) * lr_t
      # self.mom_dict.update({index: self._mom})
      # self.lr_dict.update({index: self._lr/self.lr})

  def update(self, index, weight, grad, state):
    assert (isinstance(weight, NDArray))
    assert (isinstance(grad, NDArray))
    lr = self._get_lr_dict(index)
    # lr = self._lr
    # momentum = self._mom
    wd = self._get_wd(index)
    momentum = self._get_mom(index)
    self._update_count(index)

    t = self._index_update_count[index]
    kwargs = {'rescale_grad': self.rescale_grad}
    if self.momentum > 0:
      kwargs['momentum'] = momentum
    if self.clip_gradient:
      kwargs['clip_gradient'] = self.clip_gradient

    # grad_norm = mx.ndarray.norm(grad).asscalar()
    if state is not None:
      mx.optimizer.sgd_mom_update(weight, grad, state[0], out=weight,
                                  lr=lr, wd=wd, **kwargs)
      self.after_apply(index, grad, state, t-1, lr, momentum)
      # if index == 1:
      #   print self._grad_var, self._dist_to_opt, self._h_min, self._h_max
    else:
      mx.optimizer.sgd_update(weight, grad, out=weight,
                              lr=lr, wd=wd, **kwargs)

# from yellowfin import *

mnist = mx.test_utils.get_mnist()
batch_size = 100
# train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
# val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

train_iter = mx.io.MNISTIter(
		          image="data/train-images-idx3-ubyte",
			        label="data/train-labels-idx1-ubyte",
				      data_shape=(28, 28), #data_shape=(784,),
					    batch_size=batch_size, shuffle=True, flat=False, silent=False, seed=10)
val_iter = mx.io.MNISTIter(
              image="data/t10k-images-idx3-ubyte",
			        label="data/t10k-labels-idx1-ubyte",
				      data_shape=(28, 28), #data_shape=(784,),
					    batch_size=batch_size, shuffle=True, flat=False, silent=False)

data = mx.sym.var('data')
def get_lenet(data):
  # first conv layer
  conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
  tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
  pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
  # second conv layer
  conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
  tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
  pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
  # first fullc layer
  flatten = mx.sym.flatten(data=pool2)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
  tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
  # second fullc
  fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
  # softmax loss
  lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
  return lenet

def get_mlp(data):
  # The first fully-connected layer and the corresponding activation function
  fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)
  act1 = mx.sym.Activation(data=fc1, act_type="relu")

  # The second fully-connected layer and the corresponding activation function
  fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)
  act2 = mx.sym.Activation(data=fc2, act_type="relu")
  # MNIST has 10 classes
  fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)
  # Softmax with cross entropy loss
  mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
  return mlp

model = get_mlp(data)
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on CPU
model = mx.mod.Module(symbol=model, context=mx.cpu())

# initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)

model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='YFOptimizer',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
              # initializer=initializer,
              num_epoch=10)  # train for at most 10 dataset passes

test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = model.predict(test_iter)
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy for mlp
acc = mx.metric.Accuracy()
model.score(test_iter, acc)
print(acc)

# assert acc.get()[1] > 0.98