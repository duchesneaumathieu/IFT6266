from Functionals import DGRU
from Optimizers import SGD
from Utilities.Batcher import TDB
import datetime
import pickle
import theano
import theano.tensor as T
import numpy as np
import os.path
import sys

if len(sys.argv)!=10:
    print ("python time_gru_script.py <pickle name> <unfold> <batch size> <segmentation> <depth> "
           "<#grad step> <#loop> <saving delay> <alpha>")
    sys.exit()
    
name = sys.argv[1]+".pkl"
unfold = int(sys.argv[2])
batch_size = int(sys.argv[3])
segmentation = int(sys.argv[4])
depth = int(sys.argv[5])
epoch_save = int(sys.argv[6])
epoch_max = int(sys.argv[7])
saving_delay = int(sys.argv[8])
alpha = float(sys.argv[9])

if not os.path.isfile(name):
    print "Model doesn't exist"
    print "Creating model..."
    gru = DGRU(segmentation, depth)
    print "Creating batcher..."
    batcher = TDB("Mozart", (unfold, batch_size, segmentation))
    print "Creating curve..."
    curve = []
    print "Creating map..."
    map = {"model": gru, "batcher": batcher, "curve": curve}
    print "Pickling..."
    file = open(name, "wb")
    pickle.dump(map, file)
    
else:
    print "Model already exist"
    print "Unpickling..."
    theano.config.reoptimize_unpickled_function = True
    file = open(name, "rb")
    map = pickle.load(file)
    gru = map["model"]
    batcher = map["batcher"]
    curve = map["curve"]
    
print ""
print "Creating x and y..."
x = T.tensor3("x")
y = T.tensor3("y")
print "Creating unfolded expression..."
unfold_exp = gru.unfold_apply(x, unfold)
print "Creating cost expression..."
cost_exp = T.mean((unfold_exp-y)**2)
print "Creating cost function..."
cost = theano.function([x,y], cost_exp)
print "Creating SGD class..."
sgd = SGD(x, y, gru.get_parameters(), cost_exp, batcher.get_random_batch)
print "Descent..."
vx, vy = batcher.get_valid_batch()
current_cost = cost(vx,vy)
best_cost = current_cost
best_gru = gru.copy()
t1 = datetime.datetime.now()
t2 = datetime.datetime.now()
saving = False
for i in range(epoch_max):
    sgd.descent(n_batch=epoch_save, alpha=alpha)
    vx, vy = batcher.get_valid_batch()
    current_cost = cost(vx,vy)
    curve += [current_cost]
    if current_cost < best_cost:
        best_gru = gru.copy()
        best_cost = current_cost
        t2 = datetime.datetime.now()
    if (t2-t1).seconds > saving_delay:
        map["model"] = best_gru
        file = open(name, "wb")
        pickle.dump(map, file)
        file.close()
        saving = True
        t1 = datetime.datetime.now()
    sys.stdout.write('\r%dmin %dsec completed, current cost:%f vs best cost:%f, pickled = %s'%(batcher.get_time()[0],
                                                                                            batcher.get_time()[1],
                                                                                            current_cost, best_cost, 
                                                                                            saving))
    sys.stdout.flush()
    saving = False
    
print ""
print "Pickling best..."
map["model"] = best_gru
file = open(name, "wb")
pickle.dump(map, file)
file.close()