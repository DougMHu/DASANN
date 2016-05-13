"""
Neural Network Diagram
----------------------
"""
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com

# standard libraries
import numpy as np
from matplotlib import pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.tools import FigureFactory as FF
import json
import dijkstra


def plot_overlay(test_accuracy, training_accuracy, num_epochs, xmin, 
                  training_set_size, filename,
                  test_label = "Accuracy on the test data",
                  training_label = "Accuracy on the training data",
                  num_epochs2 = None):
    if (num_epochs2 == None):
        num_epochs2 = num_epochs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy/100.0 for accuracy in test_accuracy], 
            color='#2A6EA6',
            label=test_label)
    ax.plot(np.arange(xmin, num_epochs2), 
            [accuracy*100.0/training_set_size 
             for accuracy in training_accuracy], 
            color='#FFA933',
            label=training_label)
    ax.grid(True)
    ax.set_xlim([xmin, max(num_epochs,num_epochs2)])
    ax.set_xlabel('Epoch')
    #ax.set_ylim([0, 100])
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig(filename)

def plot_training_cost(training_cost, num_epochs, training_cost_xmin, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs), 
            training_cost[training_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()
    fig.savefig(filename)

def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs), 
            [accuracy/100.0 
             for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the test data')
    plt.show()
    fig.savefig(filename)

def plot_test_cost(test_cost, num_epochs, test_cost_xmin, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_cost_xmin, num_epochs), 
            test_cost[test_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([test_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    plt.show()
    fig.savefig(filename)

def plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
            [accuracy*100.0/training_set_size 
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    plt.show()
    fig.savefig(filename, bbox_inches='tight')

def plot_connections(connections, sparsity, filename="../../sparseStudies/conn.png"):
    # code taken from: http://www.astroml.org/book_figures/appendix/fig_neural_network.html
    fig = plt.figure(facecolor='w')
    ax = fig.add_axes([0, 0, 1, 1],
                      xticks=[], yticks=[])
    plt.box(False)
    arrow_kwargs = dict(head_width=0.05, fc='black')

    #------------------------------------------------------------
    # which layers should be plotted?
    layers = connections[:-1]

    # set layer spacing
    step = 2
    numLayers = len(layers)
    start = (1 - numLayers)*step/2
    end = (numLayers - 1)*step/2
    xs = range(start, end + 1, step)
    xmin = start

    # set circle radius and minimum spacing between circles
    radius = 0.1
    spacing = 8.0/29

    #------------------------------------------------------------
    # draw circles
    ys = []
    ymin = 0
    for j, layer in enumerate(layers):
        numNeurons = len(layer)
        start = (1 - numNeurons)*spacing/2
        end = (numNeurons - 1)*spacing/2
        if (start < ymin):
            ymin = start
        store_y = []
        for y in np.arange(start, end + spacing/2, spacing):
            store_y.append(y)
            draw_circle(ax, (xs[j], y), radius, plt)
        ys.append(store_y)

    #------------------------------------------------------------
    # draw connecting arrows
    for k, y_stored in enumerate(ys[:-1]):
        for i, y1 in enumerate(y_stored):
            for j, y2 in enumerate(ys[k+1]):
                if (connections[k][j][i] == 1):
                    draw_connecting_arrow(ax, (xs[k], y1), radius, (xs[k+1], y2), radius, arrow_kwargs)

    #------------------------------------------------------------
    # Add text labels
    plt.text(0, -(ymin-1),
              "Hidden Layers: sparsity = {}".format(sparsity),
              ha='center', va='top', fontsize=16)

    ax.set_aspect('equal')
    plt.xlim((xmin-1), -(xmin-1))
    plt.ylim((ymin-2), -(ymin-2))
    #plt.show()
    plt.savefig(filename, bbox_inches='tight')

# function to draw arrows
def draw_connecting_arrow(ax, circ1, rad1, circ2, rad2, arrow_kwargs):
    theta = np.arctan2(circ2[1] - circ1[1],
                       circ2[0] - circ1[0])

    starting_point = (circ1[0] + rad1 * np.cos(theta),
                      circ1[1] + rad1 * np.sin(theta))

    length = (circ2[0] - circ1[0] - (rad1 + 1.4 * rad2) * np.cos(theta),
              circ2[1] - circ1[1] - (rad1 + 1.4 * rad2) * np.sin(theta))

    ax.arrow(starting_point[0], starting_point[1],
             length[0], length[1], **arrow_kwargs)


# function to draw circles
def draw_circle(ax, center, radius, plt):
    circ = plt.Circle(center, radius, fc='none', lw=2)
    ax.add_patch(circ)


#------------------------------------------------------------
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api
def plot_weights(weights, filename="../../sparseStudies/weights.png"):
    # which layers should be plotted?
    layers = range(len(weights))
    layers = layers[1:-1]

    x = range(len(weights[1]))
    y = range(len(weights[1][0]))
    z = weights

    annotations = []
    for k in layers:
        x = range(len(z[k]))
        for n, row in enumerate(z[k]):
            y = range(len(row))
            for m, val in enumerate(row):
                if (val != 0):
                    annotations.append(
                        dict(
                            text="{:.1f}".format(val),
                            x=x[m], y=y[n],
                            xref='x1', yref='y1',
                            font=dict(color='white' if val > 0.5 else 'black',
                                size=8),
                            showarrow=False)
                        )

        colorscale = [[0, '#FFFFFF'], [1, '#000000']]  # custom colorscale
        trace = go.Heatmap(x=x, y=y, z=z[k], colorscale=colorscale, showscale=False)

        fig = go.Figure(data=[trace])
        fig['layout'].update(
            title="Weight matrix {}".format(k),
            annotations=annotations,
            xaxis=dict(ticks='', side='top'),
            # ticksuffix is a workaround to add a bit of padding
            yaxis=dict(ticks='', ticksuffix='  ',autorange="reversed"),
            width=700,
            height=700,
            autosize=False)

        fileName = plot(fig, filename=filename[:-4] + "_{}.html".format(k))
        annotations = []

# main function: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main(jsonFile, jsonFile2=None, test_label=None, training_label=None):

    # load in the JSON file
    with open(jsonFile, "r") as f:
        dictionary = json.load(f)
    
    # output file must support these in order to create figures:
    fields = ["parameters","weights","connections",
          "test_cost","test_accuracy","training_cost","training_accuracy"]
    for field in fields:
        if (dictionary.has_key(field) == False):
            print "ERROR: Insufficient JSON file"
            return

    # print the maximum test accuracy
    logFile = jsonFile[:-5]+".txt"
    with open(logFile, "r") as f:
        last = ""
        for last in f:
            pass
    if last.split(":")[0] != "max test accuracy":
        with open(logFile, "a") as f:
            f.write("max test accuracy: {}/{}\n".format(max(dictionary["test_accuracy"]),
                                                  dictionary["parameters"]["test_set_size"]))

    # plot figures for weights and connections
    weights = dictionary["weights"]
    connections = dictionary["connections"]
    parameters = dictionary["parameters"]
    sparsity = parameters["sparsity"]
    plot_weights(weights,filename=jsonFile[:-5]+"_weights.png")
    plot_connections(connections, sparsity, filename=jsonFile[:-5]+"_conn.png")

    # determines how many input neurons are connected to output neurons
    percent, connected = dijkstra.input2output(connections)
    print percent

    # extract results from simulation
    test_accuracy = dictionary["test_accuracy"]#[:29]
    training_accuracy = dictionary["training_accuracy"]#[:29]
    test_cost = dictionary["test_cost"]#[:29]
    training_cost = dictionary["training_cost"]#[:29]

    # extract parameters from simulation
    training_set_size = parameters["training_set_size"]
    num_epochs = len(test_accuracy)
    training_cost_xmin=0
    test_accuracy_xmin=0
    test_cost_xmin=0
    training_accuracy_xmin= 0

    # plots comparing costs and accuracies
    if (len(training_cost) > 0):
        plot_training_cost(training_cost, num_epochs, training_cost_xmin, 
                        filename=jsonFile[:-5]+"_train_cost.png")
    if (len(test_accuracy) > 0):
        plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin,
                        filename=jsonFile[:-5]+"_test_accur.png")
    if (len(test_cost) > 0):
        plot_test_cost(test_cost, num_epochs, test_cost_xmin,
                    filename=jsonFile[:-5]+"_test_cost.png")
    if (len(training_accuracy) > 0):
        plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size,
                            filename=jsonFile[:-5]+"_train_accur.png")
    if ( (len(test_accuracy) > 0) and (len(training_accuracy) > 0) ):
        plot_overlay(test_accuracy, training_accuracy, num_epochs,
                 min(test_accuracy_xmin, training_accuracy_xmin),
                 training_set_size, filename=jsonFile[:-5]+"_overfit.png")

    # plots comparing different simulations
    if (jsonFile2):
        # load in the JSON file
        with open(jsonFile2, "r") as f:
            dictionary2 = json.load(f)
        # extract results from simulation
        test_accuracy2 = dictionary2["test_accuracy"]
        num_epochs2 = len(test_accuracy2)
        test_set_size2 = dictionary2["parameters"]["test_set_size"]
        file1 = jsonFile.split("/")[-1][:-5]
        file2 = jsonFile2.split("/")[-1][:-5]
        # plot comparison
        plot_overlay(test_accuracy, test_accuracy2, num_epochs,
                     test_accuracy_xmin,
                     test_set_size2, filename=jsonFile[:-5]+"_"+file2+"_compare.png",
                     test_label = test_label,
                     training_label = training_label,
                     num_epochs2 = num_epochs2)


if __name__ == "__main__":
    # main plots
    compare = False
    primary = "../training/04_05_16/sparse10_hidden3.json"

    # comparison plot
    secondary = "../training/04_05_16/sparse10_hidden3.json"
    primary_label = "Accuracy for sparsity = 0.1, Log Approximation Arithmetic"
    secondary_label = "Accuracy for sparsity = 0.1"

    if (compare):
        main(primary,jsonFile2=secondary, 
              test_label = primary_label,
              training_label = secondary_label)
    else:
        main(primary)
        






















