import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.updates import *
from lasagne.nonlinearities import *

def Model(struct, seqlen):
    inp = InputLayer((None, seqlen, struct[0]))
    l = inp
    for s in struct:
        l = LSTMLayer(l, num_units=s, unroll_scan=False, precompute_input=True)
    l_shp = ReshapeLayer(l, (-1, struct[-1]))
    l_dense = DenseLayer(l_shp, num_units=struct[0], nonlinearity=linear)
    l_out = ReshapeLayer(l_dense, (-1, seqlen, struct[0]))
    return inp, l_out

class Batcher:
    def __init__(self, batch_dim, train_percentage=90):
        from Utilities.Sound import get_sound, sound_cut
        from Utilities.Pretreatment import Normalize
        data = sound_cut(get_sound("XqaJ2Ol5cC4").astype(np.float32))
        cut = data.shape[0]*train_percentage//100
        self.pre = Normalize()
        self.pre.fit(data)
        self.train_data = self.pre.cmp(data[:cut])
        self.valid_data = self.pre.cmp(data[cut:])
        self.batch_dim = batch_dim
        self.n_batch = 0
        self.train_max = self.train_data.shape[0]-batch_dim[0]-batch_dim[1]*batch_dim[2]-1
        self.starts = np.arange(0, self.train_max-batch_dim[2], batch_dim[0])
        np.random.shuffle(self.starts)
        
    def get_batch(self):
        self.n_batch += 1
        batch_dim = self.batch_dim
        start = self.starts[self.n_batch%self.starts.shape[0]]
        x_data = np.asarray([self.train_data[start+i:start+i+batch_dim[1]*batch_dim[2]].reshape((batch_dim[1], batch_dim[2])) for i in range(batch_dim[0])])
        y_data = np.asarray([self.train_data[start+i+batch_dim[2]:start+i+(batch_dim[1]+1)*batch_dim[2]].reshape((batch_dim[1], batch_dim[2])) for i in range(batch_dim[0])])
        return x_data, y_data
    
    def get_valid_batch(self):
        batch_dim = self.batch_dim
        x_data = np.asarray([self.valid_data[i:i+batch_dim[1]*batch_dim[2]].reshape((batch_dim[1], batch_dim[2])) for i in range(batch_dim[0])])
        y_data = np.asarray([self.valid_data[i+batch_dim[2]:i+(batch_dim[1]+1)*batch_dim[2]].reshape((batch_dim[1], batch_dim[2])) for i in range(batch_dim[0])])
        return x_data, y_data
        
    def get_percentage(self):
        return 100*self.n_batch*self.batch_dim[0]//self.train_max
    
    def get_params(self):
        return self.n_batch
    
    def set_params(self, params):
        self.n_batch = params
    
class Curve:
    def __init__(self, train, valid, cost, clock):
        self.train = train
        self.valid = valid
        self.cost = cost
        self.clock = clock
        self.train_curve = []
        self.valid_curve = []
        self.clock_curve = []
        self.push()
        
    def push(self):
        self.train_curve += [self.cost(self.train[0], self.train[1])]
        self.valid_curve += [self.cost(self.valid[0], self.valid[1])]
        self.clock_curve += [self.clock()]
        
    def get_params(self):
        return [self.train_curve, self.valid_curve, self.clock_curve]
    
    def set_params(self, params):
        self.train_curve, self.valid_curve, self.clock_curve = params