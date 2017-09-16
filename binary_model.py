from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
from keras.optimizers import RMSprop, SGD, Adam



def binary_accuracy_adjusted_threshold(y_true, y_pred):
	threshold=0.5
	y_pred=K.cast(K.greater(y_pred, threshold),K.floatx())
	return K.mean(K.equal(y_true, y_pred))



def create_model():
	model = Sequential()
	model.add(Dense(128, input_dim=2883, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.005), metrics=[binary_accuracy_adjusted_threshold],verbose=1)
	return model


