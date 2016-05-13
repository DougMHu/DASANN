#############################################################
# Display histograms of weight matrices
#############################################################
# Inputs: jsonFile (string)
# Computes: Chi-squared statistic relative to a Gaussian pdf
# Displays: histogram of weight matrices

import json
import numpy as np
import myNetwork2 as myNet
import mnist_loader
import matplotlib.pyplot as plt
import scipy.stats as stats

# Choose an input file to load:
jsonFile = "../networks/sparse20_hidden3.json"

# load in the JSON file
with open(jsonFile, "r") as f:
    dictionary = json.load(f)

# extract network size, weights, and biases
sizes = dictionary["parameters"]["layers"]
weights = dictionary["weights"]

# concatenate matrix of weights into one array
weight_arrays = []
for matrix in weights:
	array = []
	for row in matrix:
		array = array + row
	weight_arrays.append(array)

# remove untrained zeros
filename = jsonFile.split("/")[-1][:-5]
sparsity = int(filename.split("_")[0][6:])
sparsity = sparsity/100.0
percentToRemove = 1 - sparsity
for array in weight_arrays:
	numToRemove = int(len(array)*percentToRemove)
	for i in range(numToRemove):
		array.remove(0)
weights = [np.array(w) for w in weight_arrays]

# plot histogram and Chi-Squared statistic
for i, array in enumerate(weights):
	chi = stats.normaltest(array)[0]
	plt.hist(array, bins=75, range=(-3,3))
	plt.title(filename + ", Chi-Squared = {}".format(chi))
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	fig = plt.gcf()
	plt.show()
	#plt.savefig(filename + "[{}].png".format(i), bbox_inches='tight')






