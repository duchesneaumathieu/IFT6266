import gzip
import pickle
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
data = pickle.load(gzip.open('mnist.pkl.gz'))

train_x = data[0][0] #: matrice de train data
train_y = data[0][1] #: vecteur des train labels

valid_x = data[1][0] # : matrice de valid data
valid_y = data[1][1] #: vecteur des valid labels

test_x = data[2][0] #: matrice de test data
test_y = data[2][1] #: vecteur des test labels

train_ohy = np.zeros((train_x.shape[0], 10))

for i, j in enumerate(train_y):
    train_ohy[i, j] = 1

valid_ohy = np.zeros((valid_x.shape[0], 10))

for i, j in enumerate(valid_y):
    valid_ohy[i, j] = 1
        
test_ohy = np.zeros((test_x.shape[0], 10))

for i, j in enumerate(test_y):
    test_ohy[i, j] = 1
    
train_x_bin = (train_x > 0.5) + 0.
valid_x_bin = (valid_x > 0.5) + 0.
test_x_bin = (test_x > 0.5) + 0.

xsets = [train_x, valid_x, test_x]
xsets_bin = [train_x_bin, valid_x_bin, test_x_bin]
ysets = [train_ohy, valid_ohy, test_ohy]

def draw(im):
    im2 = im.copy()
    for i in range(im.shape[0]):
        if im2[i] < 0: im2[i] = 0
        if im2[i] > 1: im2[i] = 1
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(im2.reshape(28,28), cmap=cm.gray_r)
    
def compare(x, toz, tox):
    draw(x)
    draw(tox(toz([x]))[0])
