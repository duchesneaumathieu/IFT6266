import theano
import theano.tensor as T

class Xent:
    def __init__(self, inputs, expression):
        self.inputs = T.matrix("y")
        self.expression = -T.log(T.max(self.inputs * expression,axis=1)).mean()
        self.function = theano.function(inputs=[inputs, self.inputs], outputs=self.expression)