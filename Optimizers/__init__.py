import numpy as np
import theano
import theano.tensor as T
from theano import config



#############
#    SGD    #
#############
class SGD:
    def __init__(self, x, y, parameters, cost, batcher):
        self.x = x
        self.y = y
        self.parameters = parameters
        self.cost = cost
        self.batcher = batcher
        self.alpha = T.scalar("alpha")
        
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
        self.alpha = T.scalar("alpha")
        
        self.grad = T.grad(self.cost, self.parameters)
        self.updates = [(self.parameters[i], self.parameters[i] - self.alpha*self.grad[i]) for i in range(len(self.parameters))]
        self.gradient_step = theano.function(inputs=[x, self.alpha], updates=self.updates)
    
    def descent(self, n_batch=1, alpha=0.1):
        for i in range(n_batch):
            x = self.batcher()
            self.gradient_step(x, alpha)


#############
#  RMSPROP  #
#############
class RMSPROP:
    def __init__(self, x, y, params, cost, epsilon=1e-6):
        self.eta = T.scalar("eta")
        self.rho = T.scalar("rho")
        self.r = []
        self.updates = []
        grads = T.grad(cost=cost, wrt=params)
        for p, g in zip(params, grads):
            r = theano.shared(0.*p.get_value(), broadcastable=p.broadcastable)
            self.r += [r]
            r_update = self.rho*r + (1-self.rho)*g**2
            gradient_scaling = T.sqrt(r_update + epsilon)
            self.updates.append((r, r_update))
            self.updates.append((p, p - self.eta*g))
        self.gradient_step = theano.function(inputs=[x, y, self.eta, self.rho], updates=self.updates)
        
    def get_params(self):
        return [self.r[i].get_value() for i in range(len(self.r))]
        
    def set_params(self, params):
        for i in range(len(self.r)): self.r[i].set_value(params[i].astype(config.floatX))
        
    def descent(self, batcher, n_batch=1, eta=0.001, rho=0.9):
        for i in range(n_batch):
            x, y = batcher()
            self.gradient_step(x, y, eta, rho)
            
class uRMSPROP:
    def __init__(self, x, params, cost, epsilon=1e-6):
        self.eta = T.scalar("eta")
        self.rho = T.scalar("rho")
        grads = T.grad(cost=cost, wrt=params)
        self.updates = []
        for p, g in zip(params, grads):
            r = theano.shared(0.*p.get_value(), broadcastable=p.broadcastable)
            r_update = self.rho*r + (1-self.rho)*g**2
            gradient_scaling = T.sqrt(r_update + epsilon)
            g = g/gradient_scaling
            self.updates.append((r, r_update))
            self.updates.append((p, p - self.eta*g))
        self.gradient_step = theano.function(inputs=[x, self.eta, self.rho], updates=self.updates)

    def get_params(self):
        return [self.r[i].get_value() for i in range(len(self.r))]
        
    def set_params(self, params):
        for i in range(len(self.r)): self.r[i].set_value(params[i].astype(config.floatX))        
        
    def descent(self, batcher, n_batch=1, eta=0.001, rho=0.9):
        for i in range(n_batch):
            x = batcher()
            self.gradient_step(x, eta, rho)