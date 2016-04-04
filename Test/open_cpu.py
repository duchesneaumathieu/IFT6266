import theano
import pickle
import sys

file = open(sys.argv[1], "rb")
theano.config.reoptimize_unpickled_function = True
map = pickle.load(file)

print map["model"].forget_gate.layers[0].linearity.weight.type()
