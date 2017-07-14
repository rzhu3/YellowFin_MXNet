import mxnet as mx
import logging
import os
import time

import numpy as np
import os
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

  def __init__(self, momentum=0.0, beta=0.999, curv_win_width=20, zero_bias=True, **kwargs):
    super(YFOptimizer, self).__init__(**kwargs)
    self.momentum = momentum
    self.beta = beta
    self.curv_win_width = 20
    self.zero_bias = zero_bias
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
    momentum = zeros(weight.shape, weight.context, dtype=weight.dtype)
    grad_avg = zeros(weight.shape, weight.context, dtype=weight.dtype)
    grad_avg_squared = zeros(weight.shape, weight.context, dtype=weight.dtype)
    return momentum, grad_avg, grad_avg_squared

  def zero_debias_factor(self):
    if not self.zero_bias:
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
    grad_avg_squared[:] = self.beta * grad_avg_squared + (1.-self.beta) * square(grad)

    # grad_norm_squared = sum(grad * grad)
    grad_norm_squared = mx.ndarray.sum(grad * grad)
    # print(grad_norm_squared.shape)
    if self._grad_norm_squared is None:
      self._grad_norm_squared = grad_norm_squared
    else:
      self._grad_norm_squared += grad_norm_squared

    if self._grad_var is None:
      self._grad_var = mx.ndarray.sum(square(grad_avg))
    else:
      self._grad_var += mx.ndarray.sum(square(grad_avg))

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
    assert (isinstance(weight, NDArray))
    assert (isinstance(grad, NDArray))
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
      self.update_grad_norm_and_var(index, grad, state)
      if self.is_end_iter():
        self.after_apply()
    else:
      mx.optimizer.sgd_update(weight, grad, out=weight,
                              lr=lr, wd=wd, **kwargs)



def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.dirname(args.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
        args.model_prefix, rank))

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, required by some networks such as resnet')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--test-io', type=int, default=0,
                       help='1 means test reading speed without training')
    return train

def fit(args, network, data_loader, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """
    optim = mx.optimizer.Optimizer.create_optimizer('YFOptimizer')
    
    
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    # data iterators
    (train, val) = data_loader(args, kv)
    if args.test_io:
        tic = time.time()
        for i, batch in enumerate(train):
            for j in batch.data:
                j.wait_to_read()
            if (i+1) % args.disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
                    i, args.disp_batches*args.batch_size/(time.time()-tic)))
                tic = time.time()

        return


    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = _load_model(args, kv.rank)
        if sym is not None:
            assert sym.tojson() == network.tojson()

    # save model
    checkpoint = _save_model(args, kv.rank)

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)

    model = mx.mod.Module(
        context       = devs,
        symbol        = network
    )

    lr_scheduler  = lr_scheduler
    optimizer_params = {
            'learning_rate': lr,
            'momentum' : args.mom,
            'wd' : args.wd,
            'lr_scheduler': lr_scheduler}
            #'multi_precision': True}

    monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None

    if args.network == 'alexnet':
        # AlexNet will not converge using Xavier
        initializer = mx.init.Normal()
    else:
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)
    # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=args.top_k))

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]
    

    #model.fit(train,  # train data
    #          eval_data=val,  # validation data
    #          optimizer='YFOptimizer',  # use SGD to train
    #          optimizer_params={'learning_rate':0.1, 'momentum':0.01},  # use fixed learning rate
    #          eval_metric='acc',  # report accuracy during training
    #          batch_end_callback = mx.callback.Speedometer(args.batch_size, 100), # output progress for each 100 data batches
    #          num_epoch=10)  # train for at most 10 dataset passes


    # run
    model.fit(train,
        begin_epoch        = args.load_epoch if args.load_epoch else 0,
        num_epoch          = args.num_epochs,
        eval_data          = val,
        eval_metric        = eval_metrics,
        #kvstore            = kv,
        # optimizer          = args.optimizer,
        optimizer          = 'YFOptimizer',  # use SGD to train
        # optimizer_params   = optimizer_params,
        optimizer_params   = {'learning_rate': 1.0, 'momentum':0},
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        batch_end_callback = batch_end_callbacks,
        epoch_end_callback = checkpoint,
        allow_missing      = True,
        monitor            = monitor)
