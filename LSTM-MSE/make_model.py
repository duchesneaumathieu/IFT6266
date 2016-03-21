import numpy as np
import pickle
import theano
import sys

from Utilities.Sound import *
from Functionals import *

argv = sys.argv

if len(argv)!=5:
    print "python model.py <pickle name> <inputs size> <LSTM depth> <unfold number>"
    sys.exit()

name = argv[1]
inputs_size = int(argv[2])
depth = int(argv[3])
unfold = int(argv[4])

lstm = LSTM(inputs_size, depth)

print "Creating inputs..."
x = theano.tensor.tensor3("x")
y = theano.tensor.tensor3("y")

print "Computing LSTM's unfold expression..."
expression = lstm.unfold_train_apply(x, unfold)

print "Computing MSE cost expression..."
cost = theano.tensor.sum((expression[0] - y)**2)

print "Constructing params..."
params = [inputs_size, unfold]

print "Constructing map..."
map = {"name": name, "model": lstm, "cost": cost, "inputs": [x, y], "params": params, "curve": []}

print "Pickling map..."
file = open(name+".pkl", 'wb', -1)
pickle.dump(map, file)

print "Model created."