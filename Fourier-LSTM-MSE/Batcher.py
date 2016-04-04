import numpy as np
import pickle
from theano import config
from Utilities.Sound import *

class Batcher:
    def __init__(self, f_len=200, unfold=40, n_seq=10):
        self.f_len = f_len
        self.unfold = unfold
        self.n_seq = n_seq
        f = open("../Mozart.pkl", "rb")
        self.data = np.asarray(pickle.load(f), dtype=config.floatX)
        maximum = self.data.shape[0]/f_len*f_len
        self.data_f = FourierEncoder(self.data[:maximum], f_len)[0]
        self.train = self.data_f[:-16*20-1]
        self.valid = self.data_f[-16*20-1:] #20 last secondes
        self.min = np.min(self.train)
        self.max = np.max(self.train)
        self.time = 0
        self.starts = None
        self.step = 0
        
    def get_batch(self, batch_size, norm=True, valid=False):
        self.time += batch_size*self.unfold

        sound = self.train
        if valid: sound = self.valid
        unfold = self.unfold
        n_seq = self.n_seq
        max = sound.shape[0]-unfold*n_seq-1
        if self.starts is None: self.starts = np.random.randint(0,max,batch_size)
        if valid: self.starts = [0]
        x = np.swapaxes(np.asarray([sound[s+self.step*unfold:s+(self.step+1)*unfold] for s in self.starts]), 0,1)
        y = np.swapaxes(np.asarray([sound[s+self.step*unfold+1:s+(self.step+1)*unfold+1] for s in self.starts]), 0,1)
        if norm:
            x = 2*(x-self.min)/float(self.max-self.min)-1
            y = 2*(y-self.min)/float(self.max-self.min)-1
        self.step += 1
        return [x.astype(config.floatX), y.astype(config.floatX)]

    def reset(self):
        self.starts = None
        self.step = 0

    def get_time(self):
        sec = int(self.time/16.)
        return (sec/60, sec%60)
