import numpy as np
from collections import defaultdict

# NOTE: This Graph class creates DIRECTED edges
class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        # comment those lines out for directed acyclic graph
        self.edges[from_node].append(to_node)
        #self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance
        #self.distances[(to_node, from_node)] = distance

# input: neural network adjacency matrices
# NOTE: ASSUMES weight matrices are NUMPY.NDARRAY types!
# returns: instance of Graph class, lists of input and output nodes
def nnToGraph(interfaces):
    # nodes in graph are stored as: (interface #, node #)
    graph = Graph()
    distance = 1
    numNodes = 0

    # adds every node from every layer except the output layer
    inputs = []
    for i, interface in enumerate(interfaces):
        numInputs = interface.shape[1]
        for inputNodes in range(numInputs):
            graph.add_node((i, inputNodes))
            if (i == 0):
                inputs = inputs + [(i, inputNodes)]
            numNodes += 1

    # adds every node from the output layer
    outputs = []
    numOutputs = interfaces[-1].shape[0]
    for outputNodes in range(numOutputs):
        graph.add_node((i+1, outputNodes))
        outputs = outputs + [(i+1, outputNodes)]
        numNodes += 1
    #print graph.nodes
    #print numNodes

    # adds the edges between each layer
    for i, interface in enumerate(interfaces):
        for j, row in enumerate(interface):
            for k, col in enumerate(row):
                if col == 1:
                    #print "({},{}) -> ({},{})".format(i,k,i+1,j)
                    graph.add_edge((i,k), (i+1,j), distance)

    return graph, inputs, outputs

# performs dijkstra alg and returns 
def dijkstra(graph, initial):
    visited = {initial: 0}
    path = {}

    nodes = set(graph.nodes)

    while nodes: 
      min_node = None
      for node in nodes:
        if node in visited:
          if min_node is None:
            min_node = node
          elif visited[node] < visited[min_node]:
            min_node = node


      if min_node is None:
        break

      nodes.remove(min_node)
      current_weight = visited[min_node]

      for edge in graph.edges[min_node]:
        weight = current_weight + graph.distances[(min_node, edge)]
        if edge not in visited or weight < visited[edge]:
          visited[edge] = weight
          path[edge] = min_node

    return visited, path

# determine what percent of the input layer neurons are
# in some way connected to one or more output layer neurons
# input: neural network adjacency matrices
# returns: percentage of input neurons connected to output neurons,
#           list of connected input neurons
def input2output(adjacencies):
    # create Graph
    graph, inputs, outputs = nnToGraph(adjacencies)

    # for every input neuron, find out if it is connected
    # indirectly to ANY output neuron
    numInputs = 0
    numConnected = 0
    connected = []
    for inputNeuron in inputs:
        numInputs += 1
        visted, path = dijkstra(graph,inputNeuron)
        connectedNeurons = visted.keys()
        for outputNeuron in outputs:
            if outputNeuron in connectedNeurons:
                numConnected += 1
                connected = connected + [inputNeuron]
    percent = 0.0
    percent = (percent+numConnected)/numInputs
    return percent, connected

def main():
    # initialize toy example of adjacency matrices
    output1 = np.array([1, 1, 0])
    output2 = np.array([0, 0, 0])
    output3 = np.array([0, 0, 1])
    interface1 = np.array( [output1, output2, output3] )
    interface2 = np.array( [output1, output2, output3] )
    interfaces = [interface1, interface2]
    #print interfaces[0].shape

    # convert adjacency matrices into Graphs
    graph, inputs, outputs = nnToGraph(interfaces)

    # run dijkstra alg and print the connected nodes,
    # their distances, and their shortest paths
    visited, path = dijkstra(graph, (0,1))
    print visited
    print path
    percent, connected = input2output(interfaces)
    print percent
    print connected

if __name__ == "__main__":
    main()
