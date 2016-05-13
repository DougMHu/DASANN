#########################################################################
# Example of loading the json file and extracting the weights and biases 
#########################################################################

import json

# Choose an input file to load:
jsonFile = "sparse10_hidden4.json"

# load in the JSON file
with open(jsonFile, "r") as f:
    dictionary = json.load(f)

###########################
# extract weights matrices
###########################

# For 4 hidden layers, there are 6 layers of neurons total.
# So, there will be a list of 5 weight matrices: one for each transition between layers
weights = dictionary["weights"]


# According to Michael Nielson's notation for weights in Chapter 2,
# the weight from the 10th neuron in the 2nd layer to the 9th neuron in the 3rd layer,
# w_{9,10}^{3} would be:
weight = weights[1][8][9]
print "w_(9,10)^(3) = {}".format(weight)
# the [1] corresponds to the 2nd weight matrix in the list
# the [8] corresponds to the 9th row of the weight matrix
# the [9] corresponds to the 10th column of the weight matrix

########################
# extract biases arrays
########################

# For 4 hidden layers, there are 6 layers of neurons total
# So, there will be a list of 5 biases: one for each layer EXCEPT the input layer
biases = dictionary["biases"]

# So, the bias from the 10th neuron in the 6th layer would be:
bias = biases[4][9][0]
print "b_(10)^(6) = {}".format(bias)
# the [4] corresponds to the 5th bias array in the list
# the [9] corresponds to the 10th neuron in the bias array
# the [0] is used because all biases are stored as 1-dim arrays



