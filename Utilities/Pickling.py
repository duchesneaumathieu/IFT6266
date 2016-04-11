import pickle
from theano import config

def pickling_model(name, model, optimizer, batcher, curve):
    map = {"model": model.get_params(),
           "optimizer": optimizer.get_params(),
           "batcher": batcher.get_params(),
           "curve": curve}
    file = open(name, "wb")
    pickle.dump(map, file)
    
def pickling_parametric_model(name, map):
    file = open(name, "wb")
    pickle.dump(map, file)
    
def unpickling_model(name, model, optimizer, batcher, curve):
    file = open(name, "rb")
    map = pickle.load(file)
    model.set_params(map["model"])
    optimizer.set_params(map["optimizer"])
    batcher.set_params(map["batcher"])
    for items in map["curve"]: curve.append(items)