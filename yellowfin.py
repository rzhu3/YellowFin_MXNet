import numpy as np
import math
import mxnet as mx

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

  def __init__(self, momentum=0.0, beta=0.999, curv_win_width=20, zero_debias=True, **kwargs):
    super(YFOptimizer, self).__init__(**kwargs)
    self.momentum = momentum
    self.beta = beta
    self.curv_win_width = 20
    self.zero_debias = zero_debias
    # The following are global states for YF tuner
    # 1. Calculate grad norm for all indices
    self._grad_norm = None
    # 2. Calculate grad norm squared for all indices
    self._grad_norm_squared = None
    # 3. Update state parameters for YF after each iteration
    # a. Used in curvature estimation
    self._h_min = 0.0
    self._h_max = 0.0
    self._h_window = np.zeros(curv_win_width)
    # b. Used in grad_variance
    self._grad_var = None
    # c. Used in distance to opt. estimation
    self._grad_norm_avg = 0.0
    self._grad_norm_squared_avg = 0.0
    self._h_avg = 0.0
    self._dist_to_opt_avg = 0.0
    # For testing purpose only
    self._test_res = []

  def create_state(self, index, weight):
    momentum = mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype)
    grad_avg = mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype)
    grad_avg_squared = mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype)
    return momentum, grad_avg, grad_avg_squared

  def zero_debias_factor(self):
    if not self.zero_debias:
      return 1.0
    return 1.0 - self.beta ** (self.num_update)

  def clear_grad_norm_info(self):
    # self._grad_norm = None
    self._grad_norm_squared = None
    self._grad_var = None

  def update_grad_norm_and_var(self, index, grad, state):
    _, grad_avg, grad_avg_squared = state
    # _, grad_avg = state
    grad_avg[:] = self.beta * grad_avg + (1. - self.beta) * grad
    grad_avg_squared[:] = self.beta * grad_avg_squared + (1. - self.beta) * mx.nd.square(grad)

    # grad_norm_squared = sum(grad * grad)
    grad_norm_squared = mx.ndarray.sum(grad * grad)
    # print(grad_norm_squared.shape)
    if self._grad_norm_squared is None:
      self._grad_norm_squared = grad_norm_squared
    else:
      self._grad_norm_squared += grad_norm_squared

    if self._grad_var is None:
      self._grad_var = mx.ndarray.sum(grad_avg * grad_avg)
    else:
      self._grad_var += mx.ndarray.sum(grad_avg * grad_avg)

  def curvature_range(self):
    curv_win = self._h_window
    beta = self.beta
    curv_win[(self.num_update-1) % self.curv_win_width] = self._grad_norm_squared
    valid_end = min(self.curv_win_width, self.num_update)
    self._h_min = beta * self._h_min + (1 - beta) * curv_win[:valid_end].min()
    self._h_max = beta * self._h_max + (1 - beta) * curv_win[:valid_end].max()
    debias_factor = self.zero_debias_factor()
    return self._h_min / debias_factor, self._h_max / debias_factor

  def grad_variance(self):
    debias_factor = self.zero_debias_factor()
    self._grad_var /= -(debias_factor ** 2)
    self._grad_var += self._grad_norm_squared_avg/debias_factor
    return self._grad_var

  def dist_to_opt(self):
    beta = self.beta
    self._grad_norm_avg = beta * self._grad_norm_avg + (1 - beta) * math.sqrt(self._grad_norm_squared)
    self._dist_to_opt_avg = beta * self._dist_to_opt_avg + (1 - beta) * self._grad_norm_avg / self._grad_norm_squared_avg
    debias_factor = self.zero_debias_factor()
    return self._dist_to_opt_avg / debias_factor

  def single_step_mu_lr(self, C, D, h_min, h_max):
    coef = np.array([-1.0, 3.0, 0.0, 1.0])
    coef[2] = -(3 + D ** 2 * h_min ** 2 / 2 / C)
    roots = np.roots(coef)
    root = roots[np.logical_and(np.logical_and(np.real(roots) > 0.0,
                                               np.real(roots) < 1.0), np.imag(roots) < 1e-5)]
    assert root.size == 1
    dr = h_max / h_min
    mu_t = max(np.real(root)[0] ** 2, ((np.sqrt(dr) - 1) / (np.sqrt(dr) + 1)) ** 2)
    lr_t = (1.0 - math.sqrt(mu_t)) ** 2 / h_min
    return mu_t, lr_t

  def after_apply(self):
    beta = self.beta

    self._grad_norm_squared = self._grad_norm_squared.asscalar()
    self._grad_norm_squared_avg = self.beta * self._grad_norm_squared_avg + (1 - self.beta) * self._grad_norm_squared

    h_min, h_max = self.curvature_range()
    C = self.grad_variance().asscalar()
    D = self.dist_to_opt()
    if self.num_update > 1:
      mu_t, lr_t = self.single_step_mu_lr(C, D, h_min, h_max)
      self.momentum = beta * self.momentum + (1 - beta) * mu_t
      self.lr = beta * self.lr + (1 - beta) * lr_t
    self._test_res = [h_max, h_min, C, D, self.lr, self.momentum]
    self.clear_grad_norm_info()

  def is_end_iter(self):
    if (self.num_update == 1) and (len(self._index_update_count) == len(self.idx2name)):
      return True
    elif (self.num_update > 1) and (np.min(self._index_update_count.values()) == self.num_update):
      return True
    else:
      return False

  def update(self, index, weight, grad, state):
    assert (isinstance(weight, mx.nd.NDArray))
    assert (isinstance(grad, mx.nd.NDArray))
    lr = self._get_lr(index)
    wd = self._get_wd(index)
    momentum = self.momentum
    self._update_count(index)

    kwargs = {'rescale_grad': self.rescale_grad}
    if self.momentum > 0:
      kwargs['momentum'] = momentum
    if self.clip_gradient:
      kwargs['clip_gradient'] = self.clip_gradient

    if state is not None:
      mx.optimizer.sgd_mom_update(weight, grad, state[0], out=weight,
                                  lr=lr, wd=wd, **kwargs)
      self.update_grad_norm_and_var(index, grad*self.rescale_grad, state)
      if self.is_end_iter():
        self.after_apply()
    else:
      mx.optimizer.sgd_update(weight, grad, out=weight,
                              lr=lr, wd=wd, **kwargs)
