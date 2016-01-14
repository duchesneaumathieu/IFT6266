import theano
import theano.tensor as T

class Xent:
    def __init__(self, predictor):
        self.inputs = T.matrix("y")
        self.expression = -T.log(T.max(self.inputs * predictor.expression,axis=1)).mean()
        self.function = theano.function(inputs=[predictor.inputs, self.inputs], outputs=self.expression)