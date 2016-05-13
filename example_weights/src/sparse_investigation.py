import mnist_loader
import LogNetwork as Net
import json
import random
import numpy as np

def main(training_data, validation_data, test_data,
		filename = "../../sparseStudies/output/dumbyfile.json", 
		sparsity = 0.1,
		layers = [784, 30, 30, 30, 10],
		epochs = 30,
		mini_batch_size = 10,
		eta = 0.1,
		lmbda = 5.0,
		early_stopping_n = 10,
		conn = None):

	# Make results more easily reproducible
    #random.seed(12345678)
    #np.random.seed(12345678)

	# create sparsely connected network
	# initialize to use Log Domain Approximation Arithmetic
	net = Net.Network(layers, sparsity=sparsity, conn=conn, math=3)

	# train
	test_cost, test_accuracy, training_cost, training_accuracy \
		= net.SGD(training_data, epochs, mini_batch_size, eta,
				  lmbda=lmbda, 
				  evaluation_data=validation_data,
				  monitor_evaluation_cost = False,
				  monitor_evaluation_accuracy=True,
	    		  monitor_training_cost=False,
	    		  monitor_training_accuracy=False,
	    		  early_stopping_n=early_stopping_n)

	# only supports these simulation outputs: hyper parameters, weight matrix, 
	# connection matrix, costs, and accuracies
	parameters = {"sparsity": sparsity, "layers": layers, "epochs": epochs,
					"mini_batch_size": mini_batch_size, "eta": eta, "lmbda": lmbda,
					"early_stopping_n": early_stopping_n, "training_set_size": len(training_data),
					"validation_set_size": len(validation_data), "test_set_size": len(test_data) }
	weights = map((lambda x: x.tolist()), net.weights)
	connections = map((lambda x: x.tolist()), net.connections)
	biases = map((lambda x: x.tolist()), net.biases)
	fields = ["parameters","weights","connections", "biases",
					"test_cost","test_accuracy","training_cost","training_accuracy"]
	values = [parameters, weights, connections, biases,
					test_cost, test_accuracy, training_cost, training_accuracy]
	dictionary = pack_output(fields,values)

	# write outputs to a JSON file
	with open(filename, "w") as f:
		json.dump(dictionary, f)

	return net.connections, net.weights

def pack_output(fields,values):
	dictionary = {}
	for field, value in zip(fields,values):
		dictionary[field] = value
	return dictionary


if __name__ == "__main__":
	# load MNIST data
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	training_data = training_data
	validation_data = validation_data
	test_data = test_data

	# specify output file names
	jsonFile = "../training/04_05_16/sparse10_hidden3.json"
	comments = "Motivation: Resimulate with Log Domain Approx Arithmetic\n"

	# specify input file
	inputFile = "../networks/sparse10_hidden3.json"

	# load in the JSON file
	with open(inputFile, "r") as f:
		dictionary = json.load(f)
    
    # input file must support these in order to create figures:
	fields = ["parameters","weights","connections",
	      "test_cost","test_accuracy","training_cost","training_accuracy"]
	for field in fields:
		if (dictionary.has_key(field) == False):
			print "ERROR: Insufficient JSON file"

    # extract parameters
	weights = dictionary["weights"]
	connections = dictionary["connections"]
	parameters = dictionary["parameters"]
	sparsity = parameters["sparsity"]
	layers = parameters["layers"]
	epochs = parameters["epochs"]
	mini_batch_size = parameters["mini_batch_size"]
	eta = parameters["eta"]
	lmbda = parameters["lmbda"]
	early_stopping_n = parameters["early_stopping_n"]


	# run sparse network
	main(training_data, validation_data, test_data,
		filename = jsonFile, 
		sparsity = sparsity, #0.1,
		layers = layers, #[784, 30, 30, 30, 30, 10],
		epochs = epochs, #60,
		mini_batch_size = mini_batch_size, #1,
		eta = eta, #0.1,
		lmbda = lmbda, #0.0,
		early_stopping_n = early_stopping_n, #30)
		conn = connections)

	# load JSON output
	with open(jsonFile, "r") as f: 
		dictionary = json.load(f)
		parameters = dictionary["parameters"]

	# write comments about simulation motivations to log file
	logFile = jsonFile[:-5]+".txt"
	with open(logFile, "w") as f:
		f.write(comments)
		for parameter, value in zip(parameters.keys(),parameters.values()):
			f.write("{}: {}\n".format(parameter, value))

















