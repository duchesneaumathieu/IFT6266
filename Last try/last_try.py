import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.updates import *
from lasagne.nonlinearities import *
from Pickling import *
from Utils import *
import datetime
import pickle
import os.path
import sys

if len(sys.argv)!=11:
    print ("python last_try.py <pickle name> <batch size> <seqlen> <segmentation> <u1> <u2> "
           "<#grad step> <#loop> <saving delay> <eta>")
    sys.exit()
    
name = sys.argv[1]+".pkl"
batch_size = int(sys.argv[2])
unfold = int(sys.argv[3])
segmentation = int(sys.argv[4])
u1 = int(sys.argv[5])
u2 = int(sys.argv[6])
epoch_save = int(sys.argv[7])
epoch_max = int(sys.argv[8])
saving_delay = int(sys.argv[9])
eta = float(sys.argv[10])


print "Creating model..."
inp, out = Model([segmentation, u1, u2], unfold) #*#
print "Creating unfolded expression..."
unfold_exp = get_output(out)
print "Creating cost expression..."
y = T.tensor3("y")
cost_exp = T.mean(T.sqr(unfold_exp-y))
print "Creating cost function..."
cost = theano.function([inp.input_var,y], cost_exp)
print "Creating SGD class..."
rms = theano.function([inp.input_var, y], updates=rmsprop(cost_exp, get_all_params(out, trainable=True), eta)) #*#
print "Creating batcher..."
batcher = Batcher((batch_size, unfold, segmentation)) #*#
print "Creating curve..."
curve = Curve(batcher.get_batch(), batcher.get_valid_batch(), cost, batcher.get_percentage) #*#

print ""

if not os.path.isfile(name):
    print "Model doesn't exist"
    print "Pickling..."
    pickling_model(name, out, batcher, curve)
    
else:
    print "Model already exist"
    print "Unpickling..."
    unpickling_model(name, out, batcher, curve)
    
print ""
print "Descent..."
current_cost = curve.valid_curve[-1]
best_cost = current_cost
best_map = {"model": get_all_param_values(out),"batcher": batcher.get_params(),"curve": curve.get_params()}
t1 = datetime.datetime.now()
t2 = datetime.datetime.now()
for i in range(epoch_max):
    for k in range(epoch_save):
        x, y = batcher.get_batch()
        rms(x, y)
    curve.push()
    current_cost = curve.valid_curve[-1]
    if current_cost < best_cost:
        best_map = {"model": get_all_param_values(out),"batcher": batcher.get_params(),"curve": curve.get_params()}
        best_cost = current_cost
    t2 = datetime.datetime.now()
    if (t2-t1).seconds > saving_delay:
        print ", Pickling model..."
        pickling_parametric_model(name, best_map)
        t1 = datetime.datetime.now()
    print_var = (batcher.get_percentage(), i, current_cost, best_cost, curve.train_curve[-1])
    sys.stdout.write('\r%d%% and %d loops completed, current cost:%f vs best cost:%f, current train cost:%f'%print_var)
    sys.stdout.flush()

print ""
print "Pickling best..."
pickling_parametric_model(name, best_map)