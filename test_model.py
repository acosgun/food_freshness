import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

model = load_model('./data/my_first_model.h5')



x_test = numpy.linspace(1,10,100, endpoint=True)

y_pred = model.predict(x_test)
print y_pred

pyplot.plot(x_test,y_pred,color='red')
pyplot.show()
