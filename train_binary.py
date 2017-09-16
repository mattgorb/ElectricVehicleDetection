import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import StratifiedKFold


def train(model,X,y, batch_size, path):
	seed = 42
	np.random.seed(seed)
	k=10
	fold=1
	kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
	for train, test in kfold.split(X,y):
		model.fit(X[train], y[train], validation_data=(X[test], y[test]),nb_epoch=50,shuffle=True, batch_size=batch_size, verbose=0,callbacks=[ReduceLROnPlateau(monitor='val_binary_accuracy_adjusted_threshold',factor=0.5,patience=3,verbose=0,epsilon=1e-4,mode='max'),
		     ModelCheckpoint(monitor='val_binary_accuracy_adjusted_threshold',filepath=str(path)+'/weights_'+str(fold)+'.h5', save_best_only=True,save_weights_only=True,mode='max', verbose=0), TensorBoard(log_dir='logs'),EarlyStopping(monitor='val_binary_accuracy_adjusted_threshold', min_delta=0, patience=10, verbose=0, mode='auto')])
		fold+=1
		print "\nTest set:  "
		print "val_binary_accuracy_adjusted_threshold: "+ str(model.evaluate(X[test], y[test], verbose=1)[1])


