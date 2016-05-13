# Simulations of Deep and Sparse Artificial Neural Networks

DASANN-simulations are Python scripts that investigate the performance of Deep and Sparse Artificial Neural Networks used to classify MNIST hand-written digits. This DASANN implementation includes clash-free permutation matrices and Log-Domain Arithmetic. The scripts output histograms of weight distributions and plots of classification accuracy over epoch.

It borrows heavily from Michael Nielsen's "Neural Networks and Deep Learning" [code samples][mnielsen], and uses his [tutorials][tut] as context.

It requires Python 2.7 and the following packages:
* numpy
* plotly
* scipy
* and standard libraries...

These scripts have only been tested in OSX.

## Installation Using Pip

Open Terminal, and run:
```
$ pip install package-name-here
```
for every package in the list.

Or if you already have them installed, run:
```
$ pip install --upgrade package-name-here
```
to upgrade any packages.

## Usage

### Simulation

To run a new simulation, open `example_weights/src/sparse_investigation.py`. Under the `__main__` section, change the `jsonFile` string to the name of your output file. Then change the input parameters to the `main()` function call. For example,
```
# run sparse network
main(training_data, validation_data, test_data,
	filename = jsonFile, 
	sparsity = 0.1,
	layers = [784, 30, 30, 30, 30, 10],
	epochs = 60,
	mini_batch_size = 1,
	eta = 0.1,
	lmbda = 0.0,
	early_stopping_n = 30)
```
This means we are creating a neural network with an input layer of 784 neurons, 4 hidden layers of 30 neurons, and an output layer of 10 neurons. In any layer, only 1/10 of the neurons are connected to the next layer (These connections are uniformly randomly chosen). SGD will do "online learning" and train on all the input data 60 times over. However, if there is no improvement in classification accuracy after 30 epochs, training will terminate prematurely. The learning rate is 0.1 with no regularization.

To change between the (faster) dot product arithmetic and the (much slower) Log-Domain arithmetic, modify the Network instantiation:
```
# create sparsely connected network
net = Net.Network(layers, sparsity=sparsity, conn=conn, math=0)
```
* Setting the input flag to `math=0` will use Python's built-in (very fast) dot product.
* Setting the input flag to `math=2` will use Exact Log-Domain Arithmetic. There should be no change in classification performance between `math=0` and `math=2`.
* Setting the input flag to `math=3` will use Approximated Log-Domain Arithmetic. The approximation is implemented using a Look-up Table.

Then open Terminal and navigate to `example_weights/src/`. Run:
```
$ python sparse_investigations.py
```
The simulation can take anywhere between 7 minutes (for `math=0`) to 7 days (for `math=3`).

### Performance Plots

To produce plots of classification accuracy, open `example_weights/src/sparse_figures.py`. Under the `__main__` section, change the `primary` file to the simulation output file you wish to plot.

Then open Terminal and navigate to `example_weights/src/`. Run:
```
$ python sparse_figures.py
```
The performance plots will appear in the same directory as the input `primary` file.

### Weight Histograms

To produce histograms of trained network weights, open `example_weights/src/weights_histograms.py`. Change the input `jsonFile` to the simulation output file you wish to plot.

Then open Terminal and navigate to `example_weights/src/`. Run:
```
$ python weights_histograms.py
```
The histograms will be displayed to the screen.

### Clash-free Permutations

The `DRP/interleaver_to_matrix.py` provides helper functions for converting [Dithered Relative Prime interleavers][crozier] into adjacency matrices in the context of neural networks.

### Storing Simulation Outputs

The best simulation outputs are stored in `sparseStudies/output/03_19_16`.

## Author
Douglas Hu (douglamh@usc.edu)
Under the guidance of:
Professor Keith Chugg (chugg@usc.edu)

[mnielsen]: https://github.com/mnielsen/neural-networks-and-deep-learning
[tut]: http://neuralnetworksanddeeplearning.com/index.html
[crozier]: https://www.researchgate.net/publication/243768615_New_High-Spread_High-Distance_Interleavers_for_Turbo-Codes
