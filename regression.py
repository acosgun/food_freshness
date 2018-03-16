import numpy
import pandas
from time import time
from matplotlib import pyplot

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return numpy.exp(-0.5 * numpy.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = numpy.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, numpy.newaxis], self.centers_,
                                 self.width_, axis=1)
    

data = pandas.read_excel("./data/data1.xlsx", sheetname='Sheet1').as_matrix()
x = numpy.float32(data[2:,0])
y = numpy.float32(data[2:,1])

gauss_model = make_pipeline(GaussianFeatures(30), LinearRegression())
gauss_model.fit(x[:, numpy.newaxis], y)
y_pred = gauss_model.predict(x[:, numpy.newaxis])

pyplot.scatter(x, y, s=1)
pyplot.plot(x, y_pred, 'green', linewidth=2)
pyplot.grid(True)
pyplot.show()
