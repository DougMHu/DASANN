#############################################################
# Test weight and bias robustness to Log Domain Approximation
#############################################################
# Inputs: jsonFile (string)
# Prints: classification accuracy of network (w/ and w/o Log Domain Approx)

import json
import numpy as np
import LogNetwork as myNet
import mnist_loader
import time;

# Choose an input file to load:
jsonFile = ["../networks/sparse10_hidden4.json", "../networks/sparse20_hidden3.json",
"../networks/sparse50_hidden3.json", "../networks/sparse100_hidden3.json",
"../networks/sparse20_hidden4.json", "../networks/sparse50_hidden4.json"]

# Load evaluation data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print
for filename in jsonFile:
	# Record simulation time
	localtime = time.asctime( time.localtime(time.time()) )
	print "start: ", localtime
	print

	# load in the JSON file
	with open(filename, "r") as f:
	    dictionary = json.load(f)
	print "input: {}".format(filename)
	print

	# extract network size, weights, and biases
	sizes = dictionary["parameters"]["layers"]
	weights = dictionary["weights"]
	biases = dictionary["biases"]

	# Create the network using Real Arithmetic
	net = myNet.Network(sizes, math=0)
	net.weights = [np.array(w) for w in weights]
	net.biases = [np.array(b) for b in biases]

	# Feedforward all test inputs and calculate classification accuracy w/ Real Arithmetic
	accuracy = net.accuracy(validation_data)
	print "Without Log Domain Approximation..."
	print "Accuracy on validation data: {} / {}".format(accuracy, len(validation_data))
	print

	# Recreate network using Log Domain Approximation Arithmetic
	net = myNet.Network(sizes, math=3)
	net.weights = [np.array(w) for w in weights]
	net.biases = [np.array(b) for b in biases]

	# feedforward
	accuracy = net.accuracy(validation_data)
	localtime = time.asctime( time.localtime(time.time()) )
	print "With Log Domain Approximation..."
	print "Accuracy on validation data: {} / {}".format(accuracy, len(validation_data))
	print
	print "stop: ", localtime
	print
	print "###########################################"
	print

