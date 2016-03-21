import numpy as np
import pickle
from Utilities.Sound import *

class Batcher:
    def __init__(self, params):
        self.params = params
        f = open("../Mozart.pkl", "rb")
        self.data = np.asarray(pickle.load(f), dtype=np.int)#np.asarray(get_sound('Mozart')[98160: 10129900], dtype=np.int)
        self.min = np.min(self.data)
        self.max = np.max(self.data)
        self.n_batch = 0
        self.epoch_size = 16000.
        self.epoch_percentage = 0
        self.epoch = 0
        
    def get_batch(self, batch_size):
        self.n_batch += batch_size
        if self.n_batch > self.epoch_size:
            self.n_batch = 0
            self.epoch += 1
        self.epoch_percentage = int(100*self.n_batch/self.epoch_size)
        
        sound = self.data
        seq_lenght, n_seq = self.params
        max = sound.shape[0]-seq_lenght*(n_seq+1)
        starts = np.random.randint(0,max,batch_size)
        x = np.swapaxes(np.asarray([sound[s:s+seq_lenght*n_seq] for s in starts]).reshape((batch_size, n_seq, seq_lenght)),0,1)
        y = np.swapaxes(np.asarray([sound[s+seq_lenght:s+seq_lenght*(n_seq+1)] for s in starts]).reshape((batch_size,
                                                                                                          n_seq, seq_lenght)),0,1)
        x_hat = 2*(x-self.min)/float(self.max-self.min)-1
        y_hat = 2*(x-self.min)/float(self.max-self.min)-1
        return [x_hat, y_hat]