import numpy as np
import theano
import theano.tensor as T
from Utilities.Scores import accuracy

class SGD: #Stochastic Gradient Descent
    def __init__(self, predictor, cost):
        self.alpha = T.dscalar("lambda")
        self.n_epoch = 0
        self.last_cost = 0
        self.memory = [0 for i in range(10)]
        self.predictor = predictor
        self.cost = cost
        self._gradient_step = theano.function(inputs=[predictor.inputs, cost.inputs, self.alpha],
                                             outputs=cost.expression,
                                             updates=predictor._update_list(self.alpha, cost.expression))
    
    def _next_batch(self, X, Y, size, i):
        start = (size * i) % X.shape[0]
        end = min(start + size, X.shape[0])
        return X[start:end, :], Y[start:end, :]

    def _gradient_epoch(self, X, Y, size, alpha):
        self.n_epoch += 1
        i=0
        while(i*size < X.shape[0]):
            x, y = self._next_batch(X, Y, size, i)
            self._gradient_step(x, y, alpha)
            i += 1
            
    def _verbose_init(self):
        print("\t\t Train cost\t\t Train acc\t\t Test cost\t\t Test acc")
        
    def print_stats(self, X, Y, TX, TY):
        print(self.n_epoch, "\t\t", "%.8f" % self.cost.function(X,Y),
              "\t\t", "%.7f" % accuracy(self.predictor.predict, X, np.argmax(Y,axis=1)),
              "\t\t", "%.7f" % self.cost.function(TX,TY),
              "\t\t", "%.6f" % accuracy(self.predictor.predict, TX, np.argmax(TY,axis=1)))
    
    def set_memory(size):
        if n_epoch == 0: self.memory = [0 for i in range(size)]
            
    def gradient_descent(self, X, Y, VX, VY, alpha, batch_size, max_epoch=100, tol=-1e-3, verbose=False, test_x=None, test_ohy=None):
        if verbose: self._verbose_init()
        if verbose: self.print_stats(X,Y,test_x,test_ohy)
        if self.n_epoch == 0: self.last_cost = self.cost.function(VX, VY)
        while(self.n_epoch < len(self.memory)):
            self._gradient_epoch(X, Y, batch_size, alpha)
            if verbose: self.print_stats(X,Y,test_x,test_ohy)
            cost = self.cost.function(VX, VY)
            self.memory[self.n_epoch-1] = cost - self.last_cost
            self.last_cost = cost
        while(self.n_epoch < max_epoch):
            if np.mean(self.memory) > tol: break
            self._gradient_epoch(X, Y, batch_size, alpha)
            if verbose: self.print_stats(X,Y,test_x,test_ohy)
            cost = self.cost.function(VX, VY)
            self.memory[(self.n_epoch-1)%len(self.memory)] = cost - self.last_cost
            self.last_cost = cost
