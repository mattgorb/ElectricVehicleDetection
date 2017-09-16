import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.model_selection import train_test_split
import binary_model
import sys
from loaddata import data
from train_binary import train
from predict import predict, get_best_threshold, predict_final,predict_on_test
import time
import random
import math
import csv
from unet_model import get_unet
from unet_data_gen import data_generator
from train_unet import train_unet
from sklearn.preprocessing import MinMaxScaler, Normalizer, scale, StandardScaler



batch_size=10


def train_with_validation_set():
	d=data()
	d.common_data_load()

	#oversample  Has_EV 
	X, y, totals, totals_x=d.train_data_load_oversampled_1(proportion=.6)
	X_2,y_2=d.train_data_load_oversampled_0(X,y)



	model=binary_model.create_model()
	train(model,X,y,batch_size, 'binary_weights')
	totals_best, num_right_best,threshold_best=get_best_threshold(totals, totals_x,model,"binary_weights/", batch_size)

	model2=binary_model.create_model()
	train(model2,X_2,y_2,batch_size, 'binary_weights2')
	totals_best2, num_right_best2,threshold_best2=get_best_threshold(totals, totals_x,model2,"binary_weights2/", batch_size)


	final=d.merge_pred_datasets(totals_best,totals_best2)

	print "Totals on weighted 1"
	print  pd.crosstab(totals_best['Y'].values, totals_best['final_bool'].values, rownames=['Actual EV'], colnames=['Predicted EV'])

	print "Totals on weighted 0"
	print  pd.crosstab(totals_best2['Y'].values, totals_best2['final_bool'].values, rownames=['Actual EV'], colnames=['Predicted EV'])

	print "Totals final"
	confusion_matrix= pd.crosstab(final['Y'].values, final['final_compare'].values, rownames=['Actual EV'], colnames=['Predicted EV'])
	print confusion_matrix
	
	


	##UNet 
	model=get_unet()
	data_gen=data_generator()
	data_gen.get_ids_to_train_from_binary_model

	data_gen.get_ids_to_train_from_binary_model(totals.index.values)

	train_gen=data_gen.train_generator()
	val_gen=data_gen.valid_generator()

	train_unet(model,train_gen,val_gen, batch_size,data_gen.ids_train_split,data_gen.ids_valid_split, "unet_weights/weights.h5")

	#predict_final(model, final, batch_size, data_gen)
	return num_right_best,threshold_best,num_right_best2, threshold_best2


	





	
def predict_test(num_right_best,threshold_best,num_right_best2, threshold_best2):
	df_test = pd.read_csv('EV_test.csv', index_col=0)
	df_test=df_test.T.fillna(value=0).T

	scaler = MinMaxScaler(feature_range=(0, 1))
	df_test_scaled = scaler.fit_transform(df_test.values.T).T
	df_test_scaled = pd.DataFrame(df_test_scaled, index=df_test.index, columns=df_test.columns)

	#add features mean, std, median, scaled on the columns
	df_binary_test_unscaled=pd.merge(df_test_scaled,df_test,left_index=True, right_index=True)

	df_test['mean'] = df_test[df_test.columns[2:]].values.mean(axis=1)
	df_test['std'] = df_test[df_test.columns[2:-1]].values.std(axis=1)
	df_test['median'] = np.median(df_test[df_test.columns[2:-2]].values,axis=1)
	
	#use standard scale across columns
	scaler = StandardScaler()
	df_test_scaled_meanstd = scaler.fit_transform(df_test[df_test.columns[-3:]].values)
	df_test_scaled_meanstd = pd.DataFrame(df_test_scaled_meanstd, index=df_test.index,columns=df_test.columns[-3:])

	df_test.drop(df_test.columns[-3:], axis=1, inplace=True)	

	df_binary_train_scaled_meanstd=pd.merge(df_test,df_test_scaled_meanstd,left_index=True, right_index=True)

	model=binary_model.create_model()

	

	#predict_on_test(model, number_right, threshold, df_binary_train_scaled_meanstd, batch_size, "binary_weights/")
	mod1_pred=predict_on_test(model, num_right_best, threshold_best, df_binary_train_scaled_meanstd, batch_size, "binary_weights2/")
	mod2_pred=predict_on_test(model, num_right_best2, threshold_best2, df_binary_train_scaled_meanstd, batch_size, "binary_weights2/")

	final=pd.merge( mod1_pred[mod1_pred.columns[np.r_[0:10, 11]]],  mod2_pred[mod2_pred.columns[np.r_[1:10, 11]]],left_index=True, right_index=True)

	final['average']=final[final.columns[np.r_[0:10, 12:21]]].mean(axis=1)

	final['final_compare']=np.where((pd.to_numeric(final.final_bool_x) & pd.to_numeric(final.final_bool_y))==1, final.final_bool_x.astype('int64'),'0')# final['average']>0.35)

	condition = ((final['final_bool_x'].astype('int64') == 0) & (final['final_bool_y'].astype('int64')==0))	
	final['final_compare']=np.where(condition, '0' , final['final_compare'])

	condition = ((final['final_bool_x'].astype('int64') == 0) & (final['final_bool_y'].astype('int64')==1))	
	final['final_compare']=np.where(condition, '0' , final['final_compare'])

	condition = ((final['final_bool_x'].astype('int64') == 1) & (final['final_bool_y'].astype('int64')==0))	
	final['final_compare']=np.where(condition,(final['average']>0.15).astype('int64'), final['final_compare'].astype('int64'))

	final=final.loc[:, ['final_bool_x','final_bool_y', 'average', 'final_compare']]

	with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
	   print(final)

	model=get_unet()
	data_gen=data_generator()
	data_gen.get_ids_to_train_from_binary_model(df_test.index.values)

	predict_final(model, final, batch_size, df_test)






num_right_best,threshold_best,num_right_best2, threshold_best2=train_with_validation_set()
predict_test(num_right_best,threshold_best,num_right_best2, threshold_best2)
#predict_test(4,.4,4, .4)
