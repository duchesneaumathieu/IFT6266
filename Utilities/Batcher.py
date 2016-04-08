import numpy as np
from theano import config
from Utilities.Sound import *
from Utilities.Pretreatment import Normalize

class TDB: #Time Domain Batcher
    def __init__(self, name, batch_dim, preprocess=Normalize(), rate=16000, n_valid=128):
        self.name = name
        self.batch_dim = batch_dim
        self.preprocess = preprocess
        self.rate = rate
        self.n_valid = n_valid
        self.valid_len = (batch_dim[0]+1)*batch_dim[2]*n_valid
        
        self.sound = sound_cut(get_sound(name).astype(config.floatX))
        self.time = self.sound.shape[0]/float(rate)
        self.preprocess.fit(self.sound[:-self.valid_len])
        
        self.train = self.preprocess.cmp(self.sound[:-self.valid_len])
        self.valid = self.preprocess.cmp(self.sound[-self.valid_len:])
        
        self.n_batch = 0
        
    def get_random_batch(self):
        self.n_batch += 1
        n_seq, batch_size, seq_lenght = self.batch_dim
        sound = self.train
        max = sound.shape[0]-seq_lenght*(n_seq+1)
        starts = np.random.randint(0,max,batch_size)
        x = np.swapaxes(np.asarray([sound[s:s+seq_lenght*n_seq] for s in starts]).reshape((batch_size, n_seq, seq_lenght)),0,1)
        y = np.swapaxes(np.asarray([sound[s+seq_lenght:s+seq_lenght*(n_seq+1)] for s in starts]).reshape((batch_size,
                                                                                                          n_seq, seq_lenght)),0,1)
        return x, y
        
    def get_valid_batch(self):
        n_seq, batch_size, seq_lenght = self.batch_dim
        sound = self.valid
        starts = np.arange(0, n_seq*seq_lenght*self.n_valid, n_seq*seq_lenght)
        x = np.swapaxes(np.asarray([sound[s:s+seq_lenght*n_seq] for s in starts]).reshape((self.n_valid, n_seq, seq_lenght)),0,1)
        y = np.swapaxes(np.asarray([sound[s+seq_lenght:s+seq_lenght*(n_seq+1)] for s in starts]).reshape((self.n_valid,
                                                                                                          n_seq, seq_lenght)),0,1)
        return x, y
    
    def get_time(self):
        n_unit = self.n_batch*self.batch_dim[0]*self.batch_dim[1]*self.batch_dim[2]
        sec = int(n_unit/float(self.rate))
        return sec/3600, (sec/60)%60, sec%60
    
class FDB: #Frequency Domain Batcher
    def __init__(self, name, batch_dim, preprocess=Normalize(), rate=16000, n_valid=128):
        self.name = name
        self.batch_dim = batch_dim
        self.preprocess = preprocess
        self.rate = rate
        self.n_valid = n_valid
        self.valid_len = (batch_dim[0]+1)*n_valid
        self.n_frequency = batch_dim[2]/2*2 + batch_dim[2] - 1
        
        self.sound = FourierEncoder(sound_cut(get_sound(name).astype(config.floatX)), self.n_frequency)[0].astype(config.floatX)
        self.time = self.sound.shape[0]/float(rate)
        self.preprocess.fit(self.sound[:-self.valid_len])
        
        self.train = self.preprocess.cmp(self.sound[:-self.valid_len])
        self.valid = self.preprocess.cmp(self.sound[-self.valid_len:])
        
        self.n_batch = 0
        
    def get_random_batch(self):
        self.n_batch += 1
        n_seq, batch_size, seq_lenght = self.batch_dim
        sound = self.train
        max = sound.shape[0]-n_seq+1
        starts = np.random.randint(0,max,batch_size)
        x = np.swapaxes(np.asarray([sound[s:s+n_seq] for s in starts]),0,1)
        y = np.swapaxes(np.asarray([sound[s+1:s+(n_seq+1)] for s in starts]),0,1)
        return x, y
        
    def get_valid_batch(self):
        n_seq, batch_size, seq_lenght = self.batch_dim
        sound = self.valid
        starts = np.arange(0, n_seq*self.n_valid, n_seq)
        x = np.swapaxes(np.asarray([sound[s:s+n_seq] for s in starts]),0,1)
        y = np.swapaxes(np.asarray([sound[s+1:s+(n_seq+1)] for s in starts]),0,1)
        return x, y
    
    def get_time(self):
        n_unit = self.n_batch*self.batch_dim[0]*self.batch_dim[1]*self.n_frequency
        sec = int(n_unit/float(self.rate))
        return sec/3600, (sec/60)%60, sec%60
