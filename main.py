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
  def __init__(self, momentum=0.0, beta=0.999, curv_win_width=20, zero_bias=False, **kwargs):
    super(YFOptimizer, self).__init__(**kwargs)
    self.momentum = momentum
    self.mom_mult = {}
    self.beta = beta
    self.curv_win_width = 20
    self.zero_bias = zero_bias
    # The following are global states for YF tuner
    self._iter = 0
    self._grad_norm_squared_avg = 0.0
    # 1. Used in curvature estimation
    self._h_min = 0.0
    self._h_max = 0.0
    self._h_window = np.zeros(curv_win_width)
    # 3. Used in distance to opt. estimation
    self._grad_norm_avg = 0.0
    self._h_avg = 0.0
    self._dist_to_opt_avg = 0.0

    self.set_mom_mult({})

  def set_mom_mult(self, args_mom_mult):
    self.mom_mult = {}
    if self.sym is not None:
      attr = self.sym.attr_dict()
      for name in self.sym.list_arguments():
        if name in attr and '__mom_mult__' in attr[name]:
          self.mom_mult[name] = float(attr[name]['__mom_mult__'])
    self.mom_mult.update(args_mom_mult)

  def _get_mom(self, index):
    if index in self.mom_mult:
      mom = self.mom_mult[index]
      return mom
    else:
      return 0.0

  def create_state(self, index, weight):
    # 2. These two state tensors are used in grad variance estimation
    # grad_avg = None
    # grad_avg_squared = None
    momentum = zeros(weight.shape, weight.context, dtype=weight.dtype)
    grad_avg = zeros(weight.shape, weight.context, dtype=weight.dtype)
    grad_avg_squared = zeros(weight.shape, weight.context, dtype=weight.dtype)
    return momentum, grad_avg, grad_avg_squared

  def zero_debias_factor(self):
    if not self.zero_bias:
      return 1.0
    return 1.0 - self.beta ** (self._iter + 1)

  def curvature_range(self, grad_norm):
    curv_win = self._h_window
    beta = self.beta
    curv_win[self._iter % self.curv_win_width] = grad_norm ** 2
    valid_end = min(self.curv_win_width, self._iter + 1)
    self._h_min = beta*self._h_min + (1-beta)*curv_win[:valid_end].min()
    self._h_max = beta*self._h_max + (1-beta)*curv_win[:valid_end].max()
    if self.zero_bias:
      debias_factor = self.zero_debias_factor()
      return self._h_min/debias_factor, self._h_max/debias_factor
    else:
      return self._h_min, self._h_max

  def grad_variance(self, grad, state):
    beta = self.beta

    _, grad_avg, grad_avg_squared = state
    # _, grad_avg = state
    grad_avg[:] = beta * grad_avg + (1-beta) * grad
    grad_avg_squared[:] = beta * grad_avg_squared + (1-beta)* grad * grad

    debias_factor = self.zero_debias_factor()
    tmp1 = mx.ndarray.sum(grad_avg * grad_avg)
    tmp2 = mx.ndarray.sum(grad_avg_squared)
    # tmp2 = self._grad_norm_squared_avg
    grad_var = tmp1 / -(debias_factor**2) + tmp2 / debias_factor
    return mx.ndarray.abs(grad_var).asscalar()

  def dist_to_opt(self, grad_norm):
    beta = self.beta
    # _, _, grad_avg_squared = state
    self._grad_norm_avg = beta*self._grad_norm_avg + (1-beta)*grad_norm
    self._dist_to_opt_avg = beta*self._dist_to_opt_avg + (1-beta)*self._grad_norm_avg/self._grad_norm_squared_avg
    debias_factor = self.zero_debias_factor()
    return self._dist_to_opt_avg / debias_factor

  def single_step_mu_lr(self, C, D, h_min, h_max):
    coef = np.array([-1.0, 3.0, 0.0, 1.0])
    coef[2] = -(3 + D**2 * h_min**2 / 2 / C)
    roots = np.roots(coef)
    root = roots[np.logical_and(np.logical_and(np.real(roots) > 0.0,
      np.real(roots) < 1.0), np.imag(roots) < 1e-5) ]
    if (root.size != 1):
      print C, D, h_min, h_max
      print self.lr
      print root
      raise ValueError
    dr = h_max / h_min
    mu_t = max(np.real(root)[0]**2, ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2 )
    lr_t = (1.0 - math.sqrt(mu_t)) ** 2 / h_min
    return mu_t, lr_t

  def after_apply(self, index, grad, state, grad_norm, lr, momentum):
    beta = self.beta
    self._grad_norm_squared_avg = self._grad_norm_squared_avg*beta + grad_norm**2 * (1-beta)
    h_min, h_max = self.curvature_range(grad_norm)
    C = self.grad_variance(grad, state)
    D = self.dist_to_opt(grad_norm)
    if self._iter > 0:
      mu_t, lr_t = self.single_step_mu_lr(C, D, h_min, h_max)
      self.set_mom_mult({index: beta*momentum + (1-beta)*mu_t})
      self.set_lr_mult({index: (beta*lr+(1-beta)*lr_t)/self.lr})

  def update(self, index, weight, grad, state):
    assert(isinstance(weight, NDArray))
    assert(isinstance(grad, NDArray))
    lr = self._get_lr(index)
    wd = self._get_wd(index)
    momentum = self._get_mom(index)
    self._update_count(index)

    kwargs = {'rescale_grad': self.rescale_grad}
    if self.momentum > 0:
      kwargs['momentum'] = momentum
    if self.clip_gradient:
      kwargs['clip_gradient'] = self.clip_gradient

    grad_norm = mx.ndarray.norm(grad).asscalar()
    if state is not None:
      mx.optimizer.sgd_mom_update(weight, grad, state[0], out=weight,
                    lr=lr, wd=wd, **kwargs)
      self.after_apply(index, grad, state, grad_norm, lr, momentum)
    else:
      mx.optimizer.sgd_update(weight, grad, out=weight,
                    lr=lr, wd=wd, **kwargs)
    self._iter += 1

optim = mx.optimizer.Optimizer.create_optimizer('YFOptimizer')

mnist = mx.test_utils.get_mnist()
batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)


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

model = get_lenet(data)
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on CPU
model = mx.mod.Module(symbol=model, context=mx.cpu())

model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='YFOptimizer',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
              num_epoch=10)  # train for at most 10 dataset passes

test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = model.predict(test_iter)
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy for mlp
acc = mx.metric.Accuracy()
model.score(test_iter, acc)
print(acc)

# assert acc.get()[1] > 0.98