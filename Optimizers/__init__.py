import numpy as np
import theano
import theano.tensor as T


class uSGD: #Stochastic Gradient Descent
    def __init__(self, x, cost_expression, cost_function, update_list):
        self.alpha = T.dscalar("alpha")
        self.n_epoch = 0
        self.cost_expression = cost_expression
        self.cost_function = cost_function
        self._gradient_step = theano.function(inputs=[x, self.alpha],
                                             updates=update_list(self.alpha, cost_expression))
    
    def _next_batch(self, X, size, i):
        start = (size * i) % X.shape[0]
        end = min(start + size, X.shape[0])
        return X[start:end, :]

    def _gradient_epoch(self, X, size, alpha):
        self.n_epoch += 1
        i=0
        while(i*size < X.shape[0]):
            x = self._next_batch(X, size, i)
            self._gradient_step(x, alpha)
            i += 1
            
    def _verbose_init(self):
        print("\t\t cost\t\t")
        
    def print_stats(self, X):
        print(self.n_epoch, "\t\t", "%.8f" % self.cost_function(X))
            
    def gradient_descent(self, X, alpha, batch_size, max_epoch=100, tol=-1e-3, verbose=False):
        if verbose: self._verbose_init()
        if verbose: self.print_stats(X)
        while(self.n_epoch < max_epoch):
            self._gradient_epoch(X, batch_size, alpha)
            if verbose: self.print_stats(X)

class uSGD: #Stochastic Gradient Descent
    def __init__(self, x, cost_expression, update_list, descentbagging=None):
        self.alpha = T.dscalar("alpha")
        self.n_epoch = 0
        self.db=descentbagging
        self._gradient_step = self._gradient_step_function(x, cost_expression, update_list)
        
    def _gradient_step_function(self, x, cost_expression, update_list):
        items = update_list
        grad = T.grad(cost_expression, items)
        updates = [(items[i], items[i] - self.alpha * grad[i]) for i in range(len(items))]
        return theano.function(inputs=[x, self.alpha], updates=updates)
    
    def _next_batch(self, X, size, i):
        start = (size * i) % X.shape[0]
        end = min(start + size, X.shape[0])
        return X[start:end, :]

    def _gradient_epoch(self, X, size, alpha):
        self.n_epoch += 1
        for i in range(int(X.shape[0]/size)+1):
            x = self._next_batch(X, size, i)
            self._gradient_step(x, alpha)
            
    def descent(self, xsets, alpha, batch_size, max_epoch, early_stop=False, verbose=False):
        X = xsets[0]
        if self.db is not None and verbose: self.db.verbose_init()
        while(self.n_epoch < max_epoch):
            if self.db is not None and self.db.do(xsets, self.n_epoch, early_stop, verbose): break 
            self._gradient_epoch(X, batch_size, alpha)
        if self.n_epoch >= max_epoch:
            self.db.compute(xsets)
            self.db.print_line(self.n_epoch)
        if verbose: self.db.print_stopped()
            
class SGD: #Stochastic Gradient Descent
    def __init__(self, x, y, cost_expression, update_list, descentbagging=None):
        self.alpha = T.dscalar("alpha")
        self.n_epoch = 0
        self.db=descentbagging
        self._gradient_step = theano.function(inputs=[x, y, self.alpha], updates=update_list(self.alpha, cost_expression))
        
    def _next_batch(self, X, Y, size, i):
        start = (size * i) % X.shape[0]
        end = min(start + size, X.shape[0])
        return X[start:end, :], Y[start:end, :]

    def _gradient_epoch(self, X, Y, size, alpha):
        self.n_epoch += 1
        for i in range(int(X.shape[0]/size)+1):
            x, y = self._next_batch(X, Y, size, i)
            self._gradient_step(x, y, alpha)
            
    def descent(self, xsets, ysets, alpha, batch_size, max_epoch, early_stop=False, verbose=False):
        X = xsets[0]
        Y = ysets[0]
        if self.db is not None and verbose: self.db.verbose_init()
        while(self.n_epoch < max_epoch):
            if self.db is not None and self.db.do(xsets, ysets, self.n_epoch, early_stop, verbose): break 
            self._gradient_epoch(X, Y, batch_size, alpha)
        if self.n_epoch >= max_epoch:
            self.db.compute(xsets, ysets)
            self.db.print_line(self.n_epoch)
        if verbose: self.db.print_stopped()

class uDescentBagging:
    def __init__(self, cost_functions, stopping_cost=1, stopping_set=1, model=None,
                 saving_delay=100, stopping_delay=100, delaying_save=10, delaying_proportion=0.25, name="best.pkl"):
        self.cost_functions = cost_functions
        self.sc = stopping_cost
        self.ss = stopping_set
        self.model = model
        self.saving_delay = saving_delay
        self.stopping_delay = stopping_delay
        self.delaying_save = delaying_save
        self.delaying_proportion = delaying_proportion
        self.name = name
        self.best_cost = float("inf")
        self.costs = [[[]for j in range(3)] for i in range(max(2, len(cost_functions)))]
        
    def compute(self, xsets):
        for i in range(max(2, len(self.cost_functions))):
            for j in range(max(3, len(xsets))):
                if i < len(self.cost_functions) and j < len(xsets):
                    self.costs[i][j].append(0.+self.cost_functions[i](xsets[j]))
                else:
                    self.costs[i][j].append(float("NaN"))
                    
    def verbose_init(self):
        print("Epoch\t\t  S1-C1\t\t  S2-C1\t\t  S3-C1\t\t  S1-C2\t\t  S2-C2\t\t  S3-C2")
        
    def print_line(self, n_epoch):
        line = []
        for i in range(2):
            for j in range(3):
                if np.isnan(self.costs[i][j][-1]): line.append("----------")
                else: line.append("%.8f" % self.costs[i][j][-1])
        print(str(n_epoch)+
              "\t\t"+ str(line[0])+
              "\t"+ str(line[1])+
              "\t"+ str(line[2])+
              "\t"+ str(line[3])+
              "\t"+ str(line[4])+
              "\t"+ str(line[5])
             )
        
    def print_saved(self):
        print("--------------------------------------------------saved---------------------------------------------------")
        
    def print_stopped(self):
        print("-------------------------------------------------stopped--------------------------------------------------")
        
    def do(self, xsets, n_epoch, early_stop, verbose):
        self.compute(xsets)
        if verbose: self.print_line(n_epoch)
        if self.saving_delay > 0: self.saving_delay -= 1
        if self.stopping_delay > 0: self.stopping_delay -= 1
        if self.costs[self.sc][self.ss][-1] < self.best_cost:
            self.best_cost = self.costs[self.sc][self.ss][-1]
            self.stopping_delay = max(self.stopping_delay, int(self.delaying_proportion*n_epoch)+1)
            early_stop = False
            if self.saving_delay == 0 and self.model is not None:
                self.model.save(self.name)
                self.saving_delay = self.delaying_save
                if verbose: self.print_saved()      
        return early_stop and self.stopping_delay == 0
            
class DescentBagging:
    def __init__(self, cost_functions, stopping_cost=1, stopping_set=1, model=None,
                 saving_delay=100, stopping_delay=100, delaying_save=10, delaying_proportion=0.25, name="best.pkl"):
        self.cost_functions = cost_functions
        self.sc = stopping_cost
        self.ss = stopping_set
        self.model = model
        self.saving_delay = saving_delay
        self.stopping_delay = stopping_delay
        self.delaying_save = delaying_save
        self.delaying_proportion = delaying_proportion
        self.name = name
        self.best_cost = float("inf")
        self.costs = [[[]for j in range(3)] for i in range(max(2, len(cost_functions)))]
        
    def compute(self, xsets, ysets):
        for i in range(max(2, len(self.cost_functions))):
            for j in range(max(3, len(xsets))):
                if i < len(self.cost_functions) and j < len(xsets):
                    self.costs[i][j].append(0.+self.cost_functions[i](xsets[j], ysets[j]))
                else:
                    self.costs[i][j].append(float("NaN"))
                    
    def verbose_init(self):
        print("Epoch\t\t  S1-C1\t\t  S2-C1\t\t  S3-C1\t\t  S1-C2\t\t  S2-C2\t\t  S3-C2")
        
    def print_line(self, n_epoch):
        line = []
        for i in range(2):
            for j in range(3):
                if np.isnan(self.costs[i][j][-1]): line.append("----------")
                else: line.append("%.8f" % self.costs[i][j][-1])
        print(str(n_epoch)+
              "\t\t"+ str(line[0])+
              "\t"+ str(line[1])+
              "\t"+ str(line[2])+
              "\t"+ str(line[3])+
              "\t"+ str(line[4])+
              "\t"+ str(line[5])
             )
        
    def print_saved(self):
        print("--------------------------------------------------saved---------------------------------------------------")
        
    def print_stopped(self):
        print("-------------------------------------------------stopped--------------------------------------------------")
        
    def do(self, xsets, ysets, n_epoch, early_stop, verbose):
        self.compute(xsets, ysets)
        if verbose: self.print_line(n_epoch)
        if self.saving_delay > 0: self.saving_delay -= 1
        if self.stopping_delay > 0: self.stopping_delay -= 1
        if self.costs[self.sc][self.ss][-1] < self.best_cost:
            self.best_cost = self.costs[self.sc][self.ss][-1]
            self.stopping_delay = max(self.stopping_delay, int(self.delaying_proportion*n_epoch)+1)
            early_stop = False
            if self.saving_delay == 0 and self.model is not None:
                self.model.save(self.name)
                self.saving_delay = self.delaying_save
                if verbose: self.print_saved()      
        return early_stop and self.stopping_delay == 0