from yellowfin import *

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

initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)

model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='YFOptimizer',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
              initializer=initializer,
              num_epoch=10)  # train for at most 10 dataset passes

test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = model.predict(test_iter)
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy for mlp
acc = mx.metric.Accuracy()
model.score(test_iter, acc)
print(acc)

# assert acc.get()[1] > 0.98