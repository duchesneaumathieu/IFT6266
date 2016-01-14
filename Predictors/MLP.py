import numpy as np
import theano
import theano.tensor as T
rng = np.random

class MLP:
    def __init__(self, layers):
        self.inputs = T.matrix("x")
        self.W = [theano.shared((rng.random((layers[i],layers[i+1]))-0.5)/(layers[i]+layers[i+1])/1000,
                                name="W"+str(i)) for i in range(len(layers)-1)]
        self.b = [theano.shared((rng.random(layers[i+1])-0.5)/(layers[i]+layers[i+1])/1000,
                                name="b"+str(i)) for i in range(len(layers)-1)]
        self.expression = self._construct_expression()
        self.function = theano.function(inputs=[self.inputs], outputs=self.expression)
        
    def _construct_expression(self):
        expression = T.maximum(0, T.dot(self.inputs, self.W[0]) + self.b[0])
        for i in range(1, len(self.W)-1):
            expression = T.maximum(0, T.dot(expression, self.W[i]) + self.b[i])
        return T.dot(expression, self.W[-1]) + self.b[-1]
    
    def _update_list(self, lamda, cost):
        item = self.W + self.b
        grad = T.grad(cost, item)
        return [(item[i], item[i] - lamda * grad[i]) for i in range(len(item))]

class MLPC(MLP):
    def __init__(self, layers):
        super().__init__(layers)
        self.expression = T.nnet.softmax(self.expression)
        self.function = theano.function(inputs=[self.inputs], outputs=self.expression)
        
    def predict_proba(self, x):
        return self.function(x)
    
    def predict(self, x):
        return np.argmax(self.function(x), axis=1)
