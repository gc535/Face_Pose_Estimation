from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import caffe
import os


def LBP_MLP(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.ip1 = L.InnerProduct(n.data, num_output=100, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=60, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.ip2, in_place=True)
    n.ip3 = L.InnerProduct(n.relu2, num_output=4, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip3, n.label)
    return n.to_proto()

def Solver(modelPath):
    s = caffe_pb2.SolverParameter()
    s.random_seed = 0xCAFFE

    # Specify locations of the train and (maybe) test networks.
    s.train_net = modelPath

    s.max_iter = 50     # no. of times to update the net (training iterations)

    # EDIT HERE to try different solvers
    # solver types include "SGD", "Adam", and "Nesterov" among others.
    s.type = "Adam"

    # Set the initial learning rate for SGD.
    s.base_lr = 0.001  # EDIT HERE to try different learning rates
    # Set momentum to accelerate learning by
    # taking weighted average of current and previous updates.
    s.momentum = 0.9
    # Set weight decay to regularize and prevent overfitting
    s.weight_decay = 5e-4

    # Set `lr_policy` to define how the learning rate changes during training.
    # This is the same policy as our default LeNet.
    s.lr_policy = 'inv'
    s.gamma = 0.0001
    s.power = 0.75
    # EDIT HERE to try the fixed rate (and compare with adaptive solvers)
    # `fixed` is the simplest policy that keeps the learning rate constant.
    # s.lr_policy = 'fixed'

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 25

    # Snapshots are files used to store networks we've trained.
    # We'll snapshot every 5K iterations -- twice during training.
    s.snapshot = 50
    s.snapshot_prefix = 'LBP_MLP'

    # Train on the GPU
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    solver_path = os.path.join(os.getcwd(), 'solver.prototxt')
    with open(solver_path, 'w') as f:
        f.write(str(s))

    return solver_path


"""

def LBP_MLP(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

with open('examples/mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('examples/mnist/mnist_train_lmdb', 64)))

with open('examples/mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('examples/mnist/mnist_test_lmdb', 100)))

"""