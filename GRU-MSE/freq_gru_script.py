from Functionals import DUGRU
from Optimizers import RMSPROP
from Utilities.Batcher import FDB
from Utilities.Pickling import *
import datetime
import pickle
import theano
import theano.tensor as T
import numpy as np
import os.path
import sys

if len(sys.argv)!=10:
    print ("python time_gru_script.py <pickle name> <unfold> <batch size> <segmentation> <depth> "
           "<#grad step> <#loop> <saving delay> <eta>")
    sys.exit()
    
name = sys.argv[1]+".pkl"
unfold = int(sys.argv[2])
batch_size = int(sys.argv[3])
segmentation = int(sys.argv[4])
depth = int(sys.argv[5])
epoch_save = int(sys.argv[6])
epoch_max = int(sys.argv[7])
saving_delay = int(sys.argv[8])
eta = float(sys.argv[9])

print "Creating model..."
gru = DUGRU([segmentation]*(depth+1), noise=False) #*#
print "Creating x and y..."
x = T.tensor3("x")
y = T.tensor3("y")
print "Creating unfolded expression..."
unfold_exp = gru.unfold_apply(x, unfold)
print "Creating cost expression..."
cost_exp = T.mean((unfold_exp[-1]-y[-1])**2)
print "Creating cost function..."
cost = theano.function([x,y], cost_exp)
print "Creating SGD class..."
rms = RMSPROP(x, y, gru.get_parameters(), cost_exp) #*#
print "Creating batcher..."
batcher = FDB("Mozart", (unfold, batch_size, segmentation)) #*#
print "Creating curve..."
curve = [] #*#

print ""

if not os.path.isfile(name):
    print "Model doesn't exist"
    print "Pickling..."
    pickling_model(name, gru, rms, batcher, curve)
    
else:
    print "Model already exist"
    print "Unpickling..."
    unpickling_model(name, gru, rms, batcher, curve)
    
print ""
print "Descent..."
vx, vy = batcher.get_valid_batch()
current_cost = cost(vx,vy)
best_cost = current_cost
best_map = {"model": gru.get_params(), "optimizer": rms.get_params(),"batcher": batcher.get_params(),"curve": curve[:]}
t1 = datetime.datetime.now()
t2 = datetime.datetime.now()
for i in range(epoch_max):
    rms.descent(batcher.get_random_batch, n_batch=epoch_save, eta=eta)
    vx, vy = batcher.get_valid_batch()
    current_cost = cost(vx,vy)
    curve += [current_cost]
    if current_cost < best_cost:
        best_map = {"model": gru.get_params(), "optimizer": rms.get_params(),"batcher": batcher.get_params(),"curve": curve[:]}
        best_cost = current_cost
    t2 = datetime.datetime.now()
    if (t2-t1).seconds > saving_delay:
        print ", Pickling model..."
        pickling_parametric_model(name, best_map)
        t1 = datetime.datetime.now()
    print_var = (batcher.get_time()[0], batcher.get_time()[1], i, current_cost, best_cost)
    sys.stdout.write('\r%dmin %dsec: %d loops completed, current cost:%f vs best cost:%f'%print_var)
    sys.stdout.flush()

print ""
print "Pickling best..."
pickling_parametric_model(name, best_map)