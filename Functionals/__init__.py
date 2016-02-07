import numpy as np
import pickle
import theano
import theano.tensor as T
rng = np.random


def relu(x):
    return T.maximum(0, x)

def no_function(x):
    return x


class Model:
    def __init__(self):
        self.updatable = None
        self.others = None
        self.updatable_items = []
            
    def mset(self, updatable, others):
        self.others = others
        self.updatable = updatable
        for i in self.updatable: self.updatable_items += i.get_items()
        
    def save(self, name):
        pickle.dump([self.updatable, self.others], open(name, "wb"))
        
    def load(self, name):
        self.updatable, self.others = pickle.load(open(name, "rb"))
        for i in self.updatable: self.updatable_items += i.get_items()
        
    def mget(self):
        return self.updatable+self.others
            
    def update_list(self, lamda, cost):
        items = self.updatable_items
        grad = T.grad(cost, items)
        return [(items[i], items[i] - lamda * grad[i]) for i in range(len(items))]
    

class MLP:
    def __init__(self, layers, inputs=None, expression=None, act_hidden=relu, act_output=T.nnet.softmax):
        self.n_layers = len(layers)
        self.inputs = inputs
        if self.inputs is None: self.inputs = T.matrix("x")
        self.act_hidden = act_hidden
        self.act_output = act_output
        
        self.W = [theano.shared((rng.random((layers[i],layers[i+1]))-0.5)/(layers[i]+layers[i+1]),
                                name="W"+str(i)) for i in range(len(layers)-1)]
        self.b = [theano.shared(np.zeros(layers[i+1]),
                                name="b"+str(i)) for i in range(len(layers)-1)]
        if expression is not None:
            self.expression = self._construct_expression(expression)
            self.function = theano.function(inputs=[self.inputs], outputs=self.expression)
            self.partial_inputs = T.matrix("x")
            self.partial_expression = self._construct_expression(self.partial_inputs)
            self.partial_function = theano.function(inputs=[self.partial_inputs], outputs=self.partial_expression)
        else:
            self.expression = self._construct_expression(self.inputs)
            self.function = theano.function(inputs=[self.inputs], outputs=self.expression)
        y = T.matrix("y")
        mc = T.mean(T.neq(T.argmax(self.expression, axis=1), T.argmax(y, axis=1)))
        self.misclass = theano.function(inputs=[self.inputs, y], outputs=mc)
        
    def _construct_expression(self, expression):
        if self.n_layers==1: return self.activation_output(expression)
        for i in range(self.n_layers-2):
            expression = self.act_hidden(T.dot(expression, self.W[i]) + self.b[i])
        return self.act_output(T.dot(expression, self.W[-1]) + self.b[-1])
    
    def get_items(self):
        return self.W + self.b
    
    def update_list(self, lamda, cost):
        items = self.W + self.b
        grad = T.grad(cost, items)
        return [(items[i], items[i] - lamda * grad[i]) for i in range(len(items))]
    
    def predict_proba(self, x):
        return self.function(x)
    
    def predict(self, x):
        return np.argmax(self.function(x), axis=1)
    
    def accuracy(self, x, y):
        return 1-self.misclass(x, y)