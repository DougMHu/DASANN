#############################################################
# Example of testing weight and bias robustness to dithering
#############################################################
# Inputs: jsonFile (string), dithering (Boolean)
# Prints: classification accuracy of network (w/ and w/o dithering)

import json
import numpy as np
import myNetwork2 as myNet
import mnist_loader

# Choose an input file to load:
jsonFile = "../networks/sparse10_hidden3.json"
dithering = True

# load in the JSON file
with open(jsonFile, "r") as f:
    dictionary = json.load(f)

# extract network size, weights, and biases
sizes = dictionary["parameters"]["layers"]
weights = dictionary["weights"]
biases = dictionary["biases"]

# Create the network
net = myNet.Network(sizes)
net.weights = [np.array(w) for w in weights]
net.biases = [np.array(b) for b in biases]

# Feedforward all test inputs and calculate classification accuracy w/o dithering
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
accuracy = net.accuracy(validation_data)
print
print "Without dithering..."
print "Accuracy on validation data: {} / {}".format(accuracy, len(validation_data))

# Dither the weights and biases
if (dithering):
	net.weights = [w+np.random.normal(0,0.1,w.shape) for w in net.weights]
	net.biases = [b+np.random.normal(0,0.1,b.shape) for b in net.biases]
	# Feedforward all test inputs and calculate classification accuracy w/ dithering
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	accuracy = net.accuracy(validation_data)
	print
	print "With dithering..."
	print "Accuracy on validation data: {} / {}".format(accuracy, len(validation_data))
	print



