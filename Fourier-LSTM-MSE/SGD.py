import theano 
import theano.tensor as T
import numpy as np
import pickle
import sys
from Batcher import *

argv = sys.argv

if len(argv) != 5:
    print "python SGD.py <pickle name> <alpha> <batch size> <max epoch>"
    sys.exit()
    
name = argv[1]
alpha = float(argv[2])
batch_size = int(argv[3])
max_epoch = int(argv[4])

print "Unpickling..."
file = open(name+".pkl", 'rb')
theano.config.reoptimize_unpickled_function = True
map = pickle.load(file)

cost = map["cost"]
model = map["model"]
inputs = map["inputs"]
params = map["params"]
expression = map["expression"]

x, m, y = inputs

print "Computing cost..."
cost_func = theano.function(inputs=inputs, outputs=[cost, expression[1]])

print "Computing update..."
updates = model.cmp_grad(alpha, cost)

print "Compution gradient step..."
sgd_step = theano.function(inputs=inputs, outputs=expression[1], updates=updates)

def valid_cost(best):
    batcher.reset()
    memory = np.zeros((model.depth, batch_size, params[0]), dtype=config.floatX)
    compilation = []
    for i in range(10):
        batch = batcher.get_batch(batch_size)
        c, memory = cost_func(batch[0], memory, batch[1])
        compilation += [c]
    map["curve"] += [np.asarray(compilation)]
    time = batcher.get_time()
    sys.stdout.write('\r%d mins %d secs: valid cost = %s'%(time[0], time[1], np.asarray(compilation)[:4].tolist()))
    if np.mean(compilation) < best:
        best = np.mean(compilation)
        file = open(name+".pkl", 'wb')
        pickle.dump(map, file, -1)
        file.close()
    return best
    
print "Descent..."
batcher = Batcher()
valid = batcher.valid
best = 10000
for n in range(max_epoch):
    best = valid_cost(best)
    batcher.reset()
    memory = np.zeros((model.depth, batch_size, params[0]), dtype=config.floatX)
    for i in range(10):
        batch = batcher.get_batch(batch_size)
        memory = sgd_step(batch[0], memory, batch[1])

print ""
