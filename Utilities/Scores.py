import numpy as np

class Stats:
    def __init__(self, function):
        self.function = function
        
def accuracy(f, x, y):
    return (f(x)==y).mean()
