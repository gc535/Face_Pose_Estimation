from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import caffe
import os


def LBP_MLP(hdf5, batch_size, phase):
    n = caffe.NetSpec()
    if phase != 'inference':
        n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    else:
        #n.data = L.Input(input_param={'shape': {'dim': [batch_size, 1, n_rows, n_cols]}})
        n.data = L.Input(input_param={'shape': {'dim':[1,324]}})
    n.ip1 = L.InnerProduct(n.data, num_output=100, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=60, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.ip2, in_place=True)
    n.ip3 = L.InnerProduct(n.relu2, num_output=2, weight_filler=dict(type='xavier'))
    if phase=='train':
        n.loss = L.SoftmaxWithLoss(n.ip3, n.label)
    elif phase=='test' or 'deploy':
        n.prob = L.Softmax(n.ip3)
        #n.accuracy = L.Accuracy(n.prob, n.label)
    return n.to_proto()

def Solver(trainModel, testModel, epoch_size, train_batch_size, epochs):
    s = caffe_pb2.SolverParameter()
    s.solver_mode: CPU
    s.random_seed = 0xCAFFE

    # Specify locations of the train and (maybe) test networks.
    s.train_net = trainModel
    #s.max_iter = 50     # no. of times to update the net (training iterations)

    s.test_net.append(testModel)
    s.test_interval = 500  # Test after every 500 training iterations.
    s.test_iter.append(300) # Test on 10 batches each time we test.

    # EDIT HERE to try different solvers
    # solver types include "SGD", "Adam", and "Nesterov" among others.
    s.type = "Adam"

    # Set the initial learning rate for SGD.
    s.base_lr = 0.001  # EDIT HERE to try different learning rates      #current best shot: 0.001
    # Set momentum to accelerate learning by
    # taking weighted average of current and previous updates.
    s.momentum = 0.99
    # Set weight decay to regularize and prevent overfitting
    s.weight_decay = 5e-4

    # Set `lr_policy` to define how the learning rate changes during training.
    # This is the same policy as our default LeNet.
    s.lr_policy = 'inv'
    s.gamma = 0.0001                                                   #current best shot: 0.00001
    s.power = 0.75
    # EDIT HERE to try the fixed rate (and compare with adaptive solvers)
    # `fixed` is the simplest policy that keeps the learning rate constant.
    #s.lr_policy = 'fixed'

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 10000

    # Snapshots are files used to store networks we've trained.
    # We'll snapshot every 5K iterations -- twice during training.
    #s.snapshot = (epoch_size//train_batch_size)*epochs - 1
    #s.snapshot_prefix = 'LBP_MLP'

    # Train on the GPU
    #s.solver_mode = caffe_pb2.SolverParameter.CPU

    # Write the solver to a temporary file and return its filename.
    solver_path = os.path.join(os.getcwd(), 'solver.prototxt')
    with open(solver_path, 'w') as f:
        f.write(str(s))

    return solver_path
