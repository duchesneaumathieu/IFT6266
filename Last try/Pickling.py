import pickle
from theano import config
from lasagne.layers import *

def pickling_model(name, model, batcher, curve):
    map = {"model": get_all_param_values(model),
           "batcher": batcher.get_params(),
           "curve": curve.get_params()}
    file = open(name, "wb")
    pickle.dump(map, file)
    
def pickling_parametric_model(name, map):
    file = open(name, "wb")
    pickle.dump(map, file)
    
def unpickling_model(name, model, batcher, curve):
    file = open(name, "rb")
    map = pickle.load(file)
    set_all_param_values(model, map["model"])
    batcher.set_params(map["batcher"])
    curve.set_params(map["curve"])
