###################################################
# translate DRP interleaver to an adjacency matrix
###################################################
# import numpy library
import numpy as np

# helper functions
def gcd(a, b):
    """Return greatest common divisor using Euclid's Algorithm."""
    while b:      
        a, b = b, a % b
    return a

def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)

def lcmm(*args):
    """Return lcm of args."""   
    return reduce(lcm, args)

def adjacency(indices, length):
	"""Given a list of indices and length of the output list,
	Return a list of 1s for adjacencies and 0s for non-adjacencies."""
	return [ int(i in indices) for i in range(length) ]

def DRP_interleaver(K, w, r, s, p):
	"""Inputs:
	K = interleaver size
	w = write dither vector, must be a factor of K
	r = read dither vector, must be a factor of K
	s = RP interleaver start index, must be in [0,K-1]
	p = RP interleaver, must be relatively prime to K
	Returns DRP interleaver
	"""
	# DRP interleaver parameters
	W = len(w)
	R = len(r)
	M = lcm(W,R)

	# construct the DRP interleaver
	r_i = [ R*(i/R) + r[i%R] for i in range(K) ]
	RP_i = [ (s+i*p) % K for i in range(K) ]
	w_i = [ W*(i/W) + w[i%W] for i in range(K) ]
	DRP_i = [ r_i[RP_i[w_i[i]]] for i in range(K) ]

	return DRP_i

	# reconstruct DRP interleaver recursively
	P_i = [ (DRP_i[i] - DRP_i[i-1]) % K for i in range(M) ]
	DRP_r = []
	for i in range(K):
		if (len(DRP_r) == 0):
			DRP_r = [DRP_i[0]] # initialize I(0)
		else:
			DRP_r = DRP_r + [(DRP_r[i-1] + P_i[i%M]) % K]

def print_interleaver(inter):
	"""Checks if interleaver is valid. Prints interleaver array."""
	print
	if ( set(range(len(inter))) == set(inter) ):
		print "Valid Interleaver:\n"
	else:
		print "INVALID INTERLEAVER:\n"
	print inter
	print

def convert_to_matrix(inter, degree_i, degree_o):
	"""Inputs:
	inter = interleaver array
	degree_i = degree of an input neuron, must be a factor of inter size
	degree_o = degree of an output neuron, must be a factor of inter size
	Returns adjacency matrix"""
	# lists of neuron indices
	K = len(inter)
	neurons_i = [i/degree_i for i in inter]
	neurons_o = [neurons_i[i*degree_o:(i+1)*degree_o] for i in range(K/degree_o)]
	print
	#print "Neuron adjacencies:\n"
	#print neurons_o
	#print

	# create adjacency matrix
	matrix = [adjacency(i, K/degree_i) for i in neurons_o]
	matrix = np.array(matrix)

	# check if any 2 neurons are connected more than once
	invalid = False
	for neurons in neurons_o:
		for i in range(len(neurons)):
			neuron = neurons.pop()
			if neuron in neurons:
				invalid = True
				break

	# if it is valid, return the matrix
	if (invalid):
		print "INVALID ADJACENCY MATRIX:"
		print
	else:
		print "Valid Adjacency Matrix:"
		print
	return matrix

def print_matrix(matrix):
		print
		print('\n'.join([''.join(['{:2}'.format(item) for item in row]) 
		      for row in matrix]))
		print

if __name__ == "__main__":
	###################################################
	# Construct DRP interleaver
	###################################################
	# input parameters
	K = 90
	w = [0, 1, 2] # size must be factor of K
	r = [0, 1, 2] # size must be factor of K
	s = 0 # must be less than K
	p = 47 # must be relatively prime to K
	###################################################
	DRP_i = DRP_interleaver(K, w, r, s, p)
	print_interleaver(DRP_i)

	###################################################
	# Convert into adjacency matrix
	###################################################
	# input parameters
	# degrees must be factors of K; # neurons = K/degree
	degree_i = 5 # degree of an input neuron
	degree_o = 5 # degree of an output neuron
	###################################################
	matrix = convert_to_matrix(DRP_i, degree_i, degree_o)
	print_matrix(matrix)

###################################################
# Case where 2 nodes are connected more than once:
# maybe because 47 ~= 90/2
###################################################
K = 90
w = [2, 4, 0, 1, 3]
r = [0, 1, 2, 3, 4]
s = 0
p = 47
degree_i = 5
degree_o = 5

###################################################
# Case where parallelism = 4, node degree = 3:
# I think parallelism is mostly determined by p
# If p is small, more parallelism is achieved
# But at the cost of very deterministic adjacencies
###################################################
K = 90
w = [1, 2, 0]
r = [0, 1, 2]
s = 0
p = 7
degree_i = 3
degree_o = 3




