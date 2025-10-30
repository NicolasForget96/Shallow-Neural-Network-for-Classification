import numpy as np

class Scaler:

    def __init__(self, method='normal'):
        self.__max = np.array([])
        self.__min = np.array([])
        self.__mean = np.array([])
        self.__std = np.array([])
        self.__method = method

        accepted_method = ['max', 'mean', 'normal', 'none']
        if method not in accepted_method:
            print('Warning: method used not supported. Will use z-normalisation (normal) instead.')
            self.__method = 'normal'

    def fit(self, X):
        self.__max = np.amax(X, axis=0)
        self.__min = np.amin(X, axis=0)
        self.__mean = np.mean(X, axis=0)
        self.__std = np.std(X, axis=0)

    def transform(self, X):
        X_scaled = np.full(X.shape, 0.0)
        
        match self.__method:
            case 'max':
                for j in range(X.shape[1]):
                    X_scaled[:, j] = X[:, j] / self.__max[j]
            case 'mean':
                for j in range(X.shape[1]):
                    X_scaled[:, j] = (X[:, j] - self.__mean[j]) / (self.__max[j] - self.__min[j])
            case 'normal':
                for j in range(X.shape[1]):
                    X_scaled[:, j] = (X[:, j] - self.__mean[j]) / self.__std[j]
            case 'none':
                X_scaled = np.copy(X) 
        
        return X_scaled