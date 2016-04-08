import numpy as np

class Normalize:
    def __init__(self):
        self.min = None
        self.max = None
    
    def fit(self, x):
        self.min = np.min(x)
        self.max = np.max(x)
    
    def cmp(self, x):
        return 2*(x-self.min)/(self.max-self.min)-1
    
    def uncmp(self, x):
        return (x+1)/2*(self.max-self.min)+self.min

class nPCA:
    def __init__(self):
        self.mean = None
        self.var = None
        self.where_enough_var = None
        self.useless_features_mean = None
        self.cov = None
        self.eig_value = None
        self.eig_vector = None
        self.eig_value_r = None
        self.eig_vector_r = None
        self.iso = None
        self.iso_inv = None
        
    def train(self, X, var_tol=0.01, eig_tol=0.01):
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)
        self.useless_features_mean = self.mean.copy()
        self.where_enough_var = np.where(np.abs(self.var) > var_tol)[0]
        self.useless_features_mean[self.where_enough_var] = 0
        self.var = self.var[self.where_enough_var]
        self.mean = self.mean[self.where_enough_var]
        X = self._delete_no_var_features(X)
        X = ((X - self.mean)/self.var).transpose()
        
        self.cov = np.cov(X)
        self.eig_value, self.eig_vector = np.linalg.eig(self.cov)
        where_enough_eig = np.where(np.absolute(self.eig_value) > eig_tol)[0]
        self.eig_value_r = np.real(self.eig_value[where_enough_eig])
        self.eig_vector_r = np.real(self.eig_vector[:, where_enough_eig])
        sortarg = self.eig_value_r.argsort()
        self.eig_value_r = self.eig_value_r[sortarg]
        self.eig_vector_r = self.eig_vector_r[:, sortarg]
        self.iso = np.dot(np.diag(1/np.sqrt(self.eig_value_r)), self.eig_vector_r.transpose())
        self.iso_inv = np.dot(self.eig_vector_r, np.diag(np.sqrt(self.eig_value_r)))
        
    def _delete_no_var_features(self, X):
        return X[:, self.where_enough_var]
        
    def transform(self, X):
        X = (self._delete_no_var_features(X) - self.mean)/self.var
        return np.dot(self.iso, X.transpose()).transpose()
    
    def transform_inv(self, Z):
        X = np.dot(self.iso_inv, Z.transpose()).transpose()
        temp = np.asarray([self.useless_features_mean.copy() for i in range(X.shape[0])])
        temp[:, self.where_enough_var] = X*self.var+self.mean
        return temp
    
    def cost(self, X):
        reconstruction = self.transform_inv(self.transform(X))
        return (np.einsum('ij,ji->i', X, reconstruction.transpose())/np.einsum('ij,ji->i', X, X.transpose())).mean()
    
    def dim(self):
        return self.iso.shape