import numpy as np
import pickle
import theano
import sys

from Utilities.Sound import *
from Functionals import *

argv = sys.argv

if len(argv)!=3:
    print "python model.py <pickle name> <LSTM depth>"
    sys.exit()

name = argv[1]
inputs_size = 101
depth = int(argv[2])
unfold = 40

lstm = LSTM(inputs_size, depth)

print "Creating inputs..."
x = theano.tensor.tensor3("x")
m = theano.tensor.tensor3("m")
y = theano.tensor.tensor3("y")

print "Computing LSTM's unfold expression..."
expression = lstm.unfold_train_apply(x, unfold, m)

print "Computing MSE cost expression..."
cost = theano.tensor.mean((expression[0] - y)**2)

print "Constructing params..."
params = [inputs_size, unfold]

print "Constructing map..."
map = {"name": name, "model": lstm, "cost": cost, "expression": expression, "inputs": [x, m, y], "params": params, "curve": []}

print "Pickling map..."
file = open(name+".pkl", 'wb', -1)
pickle.dump(map, file)

print "Model created."
