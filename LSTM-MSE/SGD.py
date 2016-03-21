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

print "Computing cost..."
cost_func = theano.function(inputs=inputs, outputs=cost)

print "Computing update..."
updates = model.cmp_grad(alpha, cost)

print "Compution gradient step..."
sgd_step = theano.function(inputs=inputs, updates=updates)

print "Descent..."
batcher = Batcher(params)
batch = batcher.get_batch(batch_size)
last_cost = cost_func(batch[0], batch[1])
best_cost = last_cost
while batcher.epoch < max_epoch:
    batch = batcher.get_batch(batch_size)
    if batcher.epoch_percentage==0:
        last_cost = cost_func(batch[0], batch[1])
        map["curve"] += [last_cost]
        if best_cost > last_cost:
            best_cost = last_cost
            file = open(name+".pkl", 'wb')
            pickle.dump(map, file, -1)
    sgd_step(batch[0], batch[1])
    sys.stdout.write('\r%d%% of epoch %d completed. Best cost: %f, last cost %f'%(batcher.epoch_percentage, batcher.epoch,
                                                                                 best_cost, last_cost))
    sys.stdout.flush()
    
print ""