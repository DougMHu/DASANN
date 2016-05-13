README
~~~~~~

Instructions:
~~~~~~~~~~~~
1. Navigate to src directory:
	$ cd example_weights/src/

2. Run test_logdomain.py script:
	$ python test_logdomain.py


Contents:
~~~~~~~~
data 	 - holds the MNIST hand-written digit training, validation, and test data

networks - JSON files containing weights and biases for different networks
		 - File naming convention: "sparse100_hidden3.json" indicates:
		 	 - sparsity = 100 (fully-connected)
		 	 - 3 hidden layers, or 5 layers of neurons total

src		 - example_load.py
			 - script explaining how to access weight and bias matrices
		 - test_robustness.py
		 	 - script that recreates a network from an input JSON file,
		 	   dithers the weights and biases,
		 	   tests the dithered network with MNIST validation data,
		 	   prints the classification accuracies w/ and w/o dithering
		 - test_logdomain.py
		 	 - similar to test_robustness, but instead of dithering weights,
		 	   it runs the classifier using Log domain operations (approximated).
		 	 - compares accuracy using Real Arithmetic vs. Log Approximated Arithmetic
		 - weights_histograms.py
		 	 - plots histogram of weights, layer by layer
		 	 - removes artificial zero weights due to sparsity
		 	 - shows gaussian distribution for fully-connected networks,
		 	   but uniform distribution with large mass at 0 for sparse networks
		 - sparse_figures.py
		 - sparse_investigation.py
		 - LogNetwork.py vs myNetwork2.py

output	 - test_logdomain.txt
			 - stores the output from test_logdomain runs

