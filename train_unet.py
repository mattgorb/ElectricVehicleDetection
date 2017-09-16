import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import StratifiedKFold


def train_unet(model,train_gen,val_gen, batch_size,ids_train_split,ids_valid_split, path):
	callbacks = [ReduceLROnPlateau(monitor='val_dice_coeff',factor=0.1,patience=10, verbose=1,epsilon=1e-4,mode='max'),
		     ModelCheckpoint(monitor='val_dice_coeff',filepath=path,save_best_only=True,save_weights_only=True,mode='max', verbose=1),EarlyStopping(monitor='val_dice_coeff', min_delta=0, patience=50, verbose=0, mode='auto')]

	model.fit_generator(generator=train_gen,
		            steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
		            epochs=100,
		            verbose=0,
		            callbacks=callbacks,
		            validation_data=val_gen,
		            validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
