import numpy as np
from collections import defaultdict

#interface1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
#interface2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
output1 = np.array([1, 0, 0])
output2 = np.array([0, 1, 0])
output3 = np.array([0, 0, 1])
interface1 = np.array( [output1, output2, output3] )
interface2 = np.array( [output1, output2, output3] )
interfaces = [interface1, interface2]
print interfaces[0].shape


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance

graph = Graph()
distance = 1
nodeId = 0
for i, interface in enumerate(interfaces):
    for inputNodes in range(interface.shape[1]):
        graph.add_node((i,inputNodes))
        nodeId += 1
print i

for outputNodes in range(interfaces[-1].shape[0]):
    graph.add_node((i+1,outputNodes))
    nodeId += 1

print graph.nodes
#for interface in interfaces:
#    for i, 


def networkToGraph(interfaces):
    pass

def dijsktra(graph, initial):
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
        weight = current_weight + graph.distance[(min_node, edge)]
        if edge not in visited or weight < visited[edge]:
          visited[edge] = weight
          path[edge] = min_node

    return visited, path