# Changes made (2/24):
# early stopping
# imported math
# added sparse_weight_initializer()
# added self.connections
# changed update_mini_batch()


"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys
import math

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost, sparsity=1, conn = None):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        if ((sparsity > 0) and (sparsity < 1)):
            if (conn):
                self.sparse_weight_initializer(sparsity, conn=conn)
            else:
                self.sparse_weight_initializer(sparsity)
        else:
            self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.connections = [np.ones_like(w) for w in self.weights]

    def sparse_weight_initializer(self, sparsity, conn = None):
        """Initialize the weights via the default_weight_initializer,
        then uniform randomly choose a few weights to keep nonzero,
        based on the sparsity factor. All other weights are zeroed out.
        """
        self.default_weight_initializer()
        # if provided connection matrices:
        if (conn):
            self.connections = [np.array(c) for c in conn]
        
        # randomly determine which network connections will be severed
        # for every layer of weights
        else:
            for layer in range(len(self.connections)):
                connections = self.connections[layer]
                # number of nonzero weights allowed in any row of weight matrix
                numNonzero = math.trunc(sparsity*self.sizes[layer])
                numZero = self.sizes[layer] - numNonzero
                # for every row in weight matrix
                for row in range(len(connections)):
                    connection = connections[row]
                    indices = range(len(connection))
                    # choose numNonzero number of unique weights to be nonzero
                    for iteration in range(numNonzero):
                        index = np.random.choice(indices)
                        indices.remove(index)
                    # then zero out the remaining weights
                    for index in indices:
                        connection[index] = 0
        
        # sever connections
        self.weights = [w*c for w, c in zip(self.weights, self.connections)]

        # print new weights
        #for index in range(len(self.weights)):
        #    A = self.weights[index]
        #    print "interface {}:".format(index)
        #    print('\n'.join([''.join(['{:6.2f}'.format(item) for item in row]) for row in A]))
        #    print

    def LDPC_weight_initializer(self, numParallel, conn = None):
        """Initialize the weights via the default_weight_initializer,
        then create an adjacency matrix that mimics the IEEE Cyclic Permutation 
        format. numParallel should be a factor of number of nodes.
        """
        self.default_weight_initializer()
        # if provided adjacency matrices, copy them
        if (conn):
            self.connections = [c for c in conn]
        # otherwise, create new adjacency matrices
        else:
            for i, layer in enumerate(self.connections):
                # calculate how many groups of parallel neurons there will be
                numSeries = len(layer) / numParallel
                numInputGroups = len(layer[0]) / numParallel
                # check if numParallel is factor of number of neurons in each layer
                numExtraRow = len(layer) % numParallel
                numExtraCol = len(layer[0]) % numParallel
                # initialize a new adjacency matrix
                layer = np.array([])
                # for each group of parallel neurons
                for j in range(numSeries):
                    # initialize a matrix of numParallel rows
                    parallelBlock = np.array([])
                    # create numInputGroups number of shifted identities
                    for k in range(numInputGroups):
                        parallelBlock = append_shifted_identity(parallelBlock, numParallel)
                    # if there are a non-integer number of input groups, add a truncated shifted identity
                    if (numExtraCol != 0):
                        parallelBlock = append_shifted_identity(parallelBlock, numParallel, 
                            trunc=numExtraCol)
                    # append parallelBlock to adjacency matrix
                    layer = append_parallel_block(layer, parallelBlock)
                # if there are a non-integer number of parallel blocks, add a truncated parallel block
                if (numExtraRow != 0):
                    parallelBlock = np.array([])
                    for k in range(numInputGroups):
                        parallelBlock = append_shifted_identity(parallelBlock, numParallel)
                    if (numExtraCol != 0):
                        parallelBlock = append_shifted_identity(parallelBlock, numParallel, 
                            trunc=numExtraCol)                   
                    layer = append_parallel_block(layer, parallelBlock, trunc=numExtraRow)
                self.connections[i] = layer

        # sever connections
        #self.weights = [w*c for w, c in zip(self.weights, self.connections)]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n=0):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        best_accuracy, last_improvement = 0, 0
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data)
            print
            # add early stopping mechanism: for no-improvement-in-n epochs
            if early_stopping_n:
                if (self.accuracy(evaluation_data) > best_accuracy):
                    best_accuracy = self.accuracy(evaluation_data)
                    last_improvement = 0
                else:
                    last_improvement += 1
                print "last improvement in accuracy: {} epochs ago".format(
                    last_improvement)
                print
                if (early_stopping_n <= last_improvement):
                    break

        # print new weights
        #for index in range(len(self.weights)):
        #    A = self.weights[index]
        #    print "interface {}:".format(index)
        #    print('\n'.join([''.join(['{:6.2f}'.format(item) for item in row]) for row in A]))
        #    print

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [((1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw)*c
                        for w, nw, c in zip(self.weights, nabla_w, self.connections)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def append_shifted_identity(original, n, trunc=None):
    """Create an n-dim identity matrix, apply a random shift, and concatenate to original
    matrix. If trunc is specified, the shifted identity columns will be truncated to this
    length. trunc must be an integer.

    """
    # create shifted identity matrix
    identity = np.identity(n)
    randomShift = np.random.choice(range(n))
    shiftedIdentity = np.roll(identity, randomShift, axis=1)
    if (trunc):
        shiftedIdentity = shiftedIdentity[:,0:trunc]
    # append shifted identity to original
    if (original.size == 0):
        original = shiftedIdentity
    else:
        original = np.concatenate((original,shiftedIdentity), axis=1)
    return original

def append_parallel_block(original, block, trunc=None):
    """Concatenate parallel block matrix to original matrix. Return modified original."""
    if (trunc):
        block = block[0:trunc,:]
    if (original.size == 0):
        original = block
    else:
        original = np.concatenate((original,block), axis=0)
    return original






