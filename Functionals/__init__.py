import numpy as np
import pickle
import theano
import theano.tensor as T
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams

def relu(x):
    return T.maximum(0, x)

def no_function(x):
    return x

###################
#    Linearity    #
###################
class Linearity:
    def __init__(self, dim_in, dim_out, weight_ini="rand"):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = self.weight_ini(weight_ini)
        self.inputs = T.matrix()
        self.cmp = theano.function([self.inputs], self.apply(self.inputs))
        
    def weight_ini(self, ini):
        if ini == "zeros":
            w = theano.shared(np.zeros((self.dim_in, self.dim_out)).astype(config.floatX))
        elif ini == "iso":
            d = max(self.dim_in, self.dim_out)
            M = np.random.normal(size=(d, d))
            U = np.linalg.eig(np.dot(M.T, M))[1]
            if self.dim_in < self.dim_out:
                w = theano.shared(U[:self.dim_in, :self.dim_out].astype(config.floatX))
            else:
                w = theano.shared((U[:self.dim_in, :self.dim_out]*(np.sqrt(self.dim_in/float(self.dim_out)))).astype(config.floatX))
        else: #rand
            w = theano.shared(((np.random.random((self.dim_in, self.dim_out))-0.5)/np.sqrt(self.dim_in+self.dim_out)).astype(config.floatX))
        return w
    
    def apply(self, inputs):
        return T.dot(T.cast(inputs, config.floatX), self.weight)

###################
#       Bias      #
###################
class Bias:
    def __init__(self, dim_out, bias_ini="zeros"):
        self.dim_out = dim_out
        self.bias = self.bias_ini(bias_ini)
        self.inputs = T.matrix()
        self.cmp = theano.function([self.inputs], self.apply(self.inputs))
        
    def bias_ini(self, ini):
        if ini == "rand":
            b = theano.shared(((np.random.random(self.dim_out)-0.5)/10).astype(config.floatX))
        elif ini == "ones":
            b = theano.shared(np.ones(self.dim_out).astype(config.floatX))
        elif ini == "zeros":
            b = theano.shared(np.zeros(self.dim_out).astype(config.floatX))
        else: #uniform
            b = theano.shared((ini*np.ones(self.dim_out)).astype(config.floatX))
        return b
        
    def apply(self, inputs):
        return inputs + self.bias
    

###################
#   NonLinearity  #
###################
class NonLinearity:
    def __init__(self, non_lin):
        self.non_lin = non_lin
        self.inputs = T.matrix()
        self.cmp = theano.function([self.inputs], self.apply(self.inputs))
        
    def apply(self, inputs):
        return self.non_lin(inputs)
    
    
###################
#    BatchNorm    #
###################
class BatchNormalization:
    def __init__(self, dim_out, epsilon=1e-5):
        self.dim_out = dim_out
        self.epsilon = epsilon
        self.mean = theano.shared(np.zeros(dim_out).astype(config.floatX))
        self.var = theano.shared(np.ones(dim_out).astype(config.floatX))
        self.gamma = theano.shared(np.ones(dim_out).astype(config.floatX))
        self.inputs = T.matrix()
        self.train_cmp = theano.function([self.inputs], self.train_apply(self.inputs))
        self.cmp = theano.function([self.inputs], self.apply(self.inputs))
        
    def train_apply(self, inputs):
        mean = T.mean(inputs, axis=0)
        var = T.var(inputs, axis=0)
        return self.gamma*(inputs - mean)/T.sqrt(var+self.epsilon)
    
    def compute_parameters(self, inputs):
        self.mean.set_value(np.mean(inputs, axis=0))
        self.var.set_value(np.var(inputs, axis=0))
        
    def apply(self, inputs):
        return self.gamma*(inputs - self.mean)/T.sqrt(self.var+self.epsilon)
    
    
###################
#      Noise      #
###################
class Noise:
    def __init__(self, dim_out, std=1e-2):
        self.dim_out = dim_out
        self.std = theano.shared(std)
        self.rng = RandomStreams()
        self.inputs = T.matrix()
        self.cmp = theano.function([self.inputs], self.apply(self.inputs))
        
    def apply(self, inputs):
        return inputs + self.rng.normal(std=self.std, size=(inputs.shape[0], self.dim_out), dtype=config.floatX)
    
    
###################
#     Dropout     #
###################
class Dropout:
    def __init__(self, dim_out):
        self.dim_out = dim_out
        self.rng = RandomStreams()
        self.inputs = T.matrix()
        self.cmp = theano.function([self.inputs], self.apply(self.inputs))
        
    def apply(self, inputs):
        mask = self.rng.binomial(n=1, p=0.5, size=(inputs.shape[0], self.dim_out))
        return 2*inputs*T.cast(mask, theano.config.floatX) #multiply by 2 to keep the expected value of the norm the same
    
    
###################
#      Layer      #
###################
class Layer:
    def __init__(self, dim_in, dim_out, non_linearity, weight_ini="rand", bias_ini="zeros",
                 noise=False, batch_norm=False, dropout=False, noise_std=1e-2, batch_norm_epsilon=1e-5):
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.noise_activated = noise
        self.batch_norm_activated = batch_norm
        self.dropout_activated = dropout
        
        self.linearity = Linearity(dim_in, dim_out, weight_ini=weight_ini)
        self.batch_norm = BatchNormalization(dim_out, epsilon=batch_norm_epsilon)
        self.bias = Bias(dim_out, bias_ini=bias_ini)
        self.noise = Noise(dim_out, std=noise_std)
        self.dropout = Dropout(dim_out)
        self.non_linearity = NonLinearity(non_linearity)
        
        self.inputs = T.matrix()
        self.train_cmp = theano.function([self.inputs], self.train_apply(self.inputs))
        self.cmp = theano.function([self.inputs], self.apply(self.inputs))
        
    def compute_parameters(self, inputs):
        self.batch_norm.compute_parameters(self.linearity.cmp(inputs))
        
    def get_parameters(self):
        parameters = [self.linearity.weight] + [self.bias.bias]
        if self.batch_norm_activated: parameters += [self.batch_norm.gamma]
        return parameters
        
    def train_apply(self, inputs):
        x = self.linearity.apply(inputs)
        if self.batch_norm_activated: x = self.batch_norm.train_apply(x)
        if self.noise_activated: x = self.noise.apply(x)
        if self.dropout_activated: x = self.dropout.apply(x)
        x = self.bias.apply(x)
        x = self.non_linearity.apply(x)
        return x
        
    def apply(self, inputs):
        x = self.linearity.apply(inputs)
        if self.batch_norm_activated: x = self.batch_norm.apply(x)
        x = self.bias.apply(x)
        x = self.non_linearity.apply(x)
        return x
    
    
###################
#       MLP       #
###################
class MLP:
    def __init__(self, struct, inner_non_linearity, final_non_linearity, weight_ini="rand", bias_ini="zeros",
                 noise=False, batch_norm=False, dropout=False, noise_std=1e-2, batch_norm_epsilon=1e-5):
        assert len(struct) > 1, "struct should at least have lenght 2"
        self.depth = len(struct)-1
        self.layers = []
        for i in range(self.depth-1):
            self.layers += [Layer(struct[i], struct[i+1], inner_non_linearity,
                                  weight_ini=weight_ini, bias_ini=bias_ini,
                                  noise=noise, batch_norm=batch_norm, dropout=dropout,
                                  noise_std=noise_std, batch_norm_epsilon=batch_norm_epsilon)]
                            
        self.layers += [Layer(struct[-2], struct[-1], final_non_linearity,
                              weight_ini=weight_ini, bias_ini=bias_ini,
                              noise=noise, batch_norm=batch_norm, dropout=dropout,
                              noise_std=noise_std, batch_norm_epsilon=batch_norm_epsilon)]
        
        self.inputs = T.matrix()
        self.train_cmp = theano.function([self.inputs], self.train_apply(self.inputs))
        self.cmp = theano.function([self.inputs], self.apply(self.inputs))
        
    def get_parameters(self):
        parameters = []
        for layer in self.layers:
            parameters += layer.get_parameters()
        return parameters
            
    def cmp_grad(self, alpha, cost):
        parameters = self.get_parameters()
        grad = T.grad(cost, parameters)
        return [(parameters[i], parameters[i] - alpha * grad[i]) for i in range(len(parameters))]
    
    def set_noise_std(self, std):
        for layer in self.layers:
            layer.noise.std.set_value(std)
        
    def compute_parameters(self, inputs):
        x = inputs
        for layer in self.layers:
            layer.compute_parameters(x)
            x = layer.cmp(x)
    
    def train_apply(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.train_apply(x)
        return x
    
    def apply(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.apply(x)
        return x
    
    
###################
#       LSTM       #
###################
class LSTM:
    def __init__(self, inputs_size, depth=1):
        self.depth = depth
        self.inputs_size = inputs_size
        self.rng = RandomStreams()
        size = 2*inputs_size
        self.forget_gate = MLP([size, size, inputs_size], relu, T.nnet.sigmoid,
                          weight_ini="iso", bias_ini=2, noise=True, noise_std=5e-3)
        self.input_gate = MLP([size, size, inputs_size], relu, T.nnet.sigmoid,
                          weight_ini="iso", noise=True, noise_std=5e-3)
        self.tanh_gate = MLP([size, size, inputs_size], T.tanh, T.tanh,
                        weight_ini="iso", noise=True, noise_std=5e-3)
        self.output_gate = MLP([depth*inputs_size, size, inputs_size], relu, no_function,
                       weight_ini="rand", noise=True, noise_std=5e-3)
    
    def get_parameters(self):
        parameters = self.forget_gate.get_parameters()
        parameters += self.input_gate.get_parameters()
        parameters += self.tanh_gate.get_parameters()
        parameters += self.output_gate.get_parameters()
        return parameters

    def copy(self):
        copy = LSTM(self.inputs_size, self.depth)
        params = self.get_parameters()
        copy_params = copy.get_parameters()
        for i in range(len(params)):
            copy_params[i].set_value(params[i].get_value().copy())
        return copy
            
    def cmp_grad(self, alpha, cost):
        parameters = self.get_parameters()
        grad = T.grad(cost, parameters)
        return [(parameters[i], parameters[i] - alpha * grad[i]) for i in range(len(parameters))]
    
    
    def unfold_train_apply(self, inputs, lenght, memory=None):
        if memory is None: memory = [self.rng.normal(std=1e-2, size=(inputs[0].shape[0], self.inputs_size)) for i in range(self.depth)]
        outputs = []
        for i in range(lenght):
            y, memory = self.train_apply((inputs[i], memory))
            outputs += [y]
        return T.as_tensor_variable(outputs), T.as_tensor_variable(memory)
        
    def unfold_apply(self, inputs, lenght, memory=None):
        if memory is None: memory = [self.rng.normal(std=1e-2, size=(inputs[0].shape[0], self.inputs_size)) for i in range(self.depth)]
        outputs = []
        for i in range(lenght):
            y, memory = self.apply((inputs[i], memory))
            outputs += [y]
        return T.as_tensor_variable(outputs), T.as_tensor_variable(memory)
        
    def train_apply(self, inputs): 
        x, memory_cells = inputs
        out_memory = []
        for d in range(self.depth):
            memory = memory_cells[d]
            inputs_d = T.concatenate([memory, x], axis=1)
            forgeted_memory = memory*self.forget_gate.train_apply(inputs_d)
            x = forgeted_memory + self.input_gate.train_apply(inputs_d)*self.tanh_gate.train_apply(inputs_d)
            out_memory += [x]
        concat_memory = T.concatenate(out_memory, axis=1)
        y = self.output_gate.train_apply(concat_memory)
        return y, out_memory
        
    def apply(self, inputs): 
        x, memory_cells = inputs
        out_memory = []
        for d in range(self.depth):
            memory = memory_cells[d]
            inputs_d = T.concatenate([memory, x], axis=1)
            forgeted_memory = memory*self.forget_gate.apply(inputs_d)
            x = forgeted_memory + self.input_gate.apply(inputs_d)*self.tanh_gate.apply(inputs_d)
            out_memory += [x]
        concat_memory = T.concatenate(out_memory, axis=1)
        y = self.output_gate.apply(concat_memory)
        return y, out_memory
