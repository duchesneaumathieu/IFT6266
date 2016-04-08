import numpy as np
import theano
import theano.tensor as T


class SGD:
    def __init__(self, x, y, parameters, cost, batcher):
        self.x = x
        self.y = y
        self.parameters = parameters
        self.cost = cost
        self.batcher = batcher
        self.alpha = T.dscalar("alpha")
        
        self.grad = T.grad(self.cost, self.parameters)
        self.updates = [(self.parameters[i], self.parameters[i] - self.alpha*self.grad[i]) for i in range(len(self.parameters))]
        self.gradient_step = theano.function(inputs=[x, y, self.alpha], updates=self.updates)
    
    def descent(self, n_batch=1, alpha=0.1):
        for i in range(n_batch):
            x, y = self.batcher()
            self.gradient_step(x, y, alpha)
            
class uSGD:
    def __init__(self, x, parameters, cost, batcher):
        self.x = x
        self.parameters = parameters
        self.cost = cost
        self.batcher = batcher
        self.alpha = T.dscalar("alpha")
        
        self.grad = T.grad(self.cost, self.parameters)
        self.updates = [(self.parameters[i], self.parameters[i] - self.alpha*self.grad[i]) for i in range(len(self.parameters))]
        self.gradient_step = theano.function(inputs=[x, self.alpha], updates=self.updates)
    
    def descent(self, n_batch=1, alpha=0.1):
        for i in range(n_batch):
            x = self.batcher()
            self.gradient_step(x, alpha)