import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler, Normalizer, scale, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from sklearn.utils import shuffle

class data_generator:
	batch_size=20
	df_train_scaled=[]
	df_labels=[]
	ids_train_split=[]
	ids_valid_split=[]

	def __init__(self):
		global batch_size, df_train_scaled, ids_train_split, ids_valid_split
		df_train = pd.read_csv('EV_train.csv', index_col=0)
		df_train=df_train.T.fillna(value=0).T
		houses_w_nans=df_train[pd.isnull(df_train).any(axis=1)]
		houses_missing_data=houses_w_nans.index.tolist()
		df_train=df_train[~df_train.index.isin(houses_missing_data)]

		#try MinMax and StandardScalar
		scaler = MinMaxScaler(feature_range=(0, 1))
		self.df_train_scaled = scaler.fit_transform(df_train.values.T).T
		self.df_train_scaled = pd.DataFrame(self.df_train_scaled, index=df_train.index, columns=df_train.columns)

		#labels data
		self.df_labels=pd.read_csv('EV_train_labels.csv', index_col=0)
		ev_label_all=self.df_labels.max(axis=1).rename('Has_EV')

		ev_label_1=ev_label_all.loc[ev_label_all.values==1]
		ev_label_0=ev_label_all.loc[ev_label_all.values==0]

		#to train on 1s 
		df_train_1=df_train[self.df_train_scaled.index.isin(ev_label_1.index)]
		ids_train_1 =df_train_1.index.values

		#train on 5050 data
		undersampled_labels=pd.concat([ev_label_1[:int(len(ev_label_1))], ev_label_0[:int(len(ev_label_1))]])
		undersampled_labels=undersampled_labels.index.values
		#print undersampled_labels

		ids_train=df_train.index.values
		self.ids_train_split, self.ids_valid_split = train_test_split(undersampled_labels, test_size=0.2, random_state=42)

		np.set_printoptions(threshold=np.nan)


	def get_ids_to_train_from_binary_model(self, test):
		self.ids_train_split, self.ids_valid_split
		
		ev_label_all=self.df_labels.max(axis=1).rename('Has_EV')

		ev_label_1=ev_label_all.loc[ev_label_all.values==1]
		ev_label_0=ev_label_all.loc[ev_label_all.values==0]

		train_set_1=ev_label_1[~ev_label_1.index.isin(test)]
		train_set_0=ev_label_0[~ev_label_0.index.isin(test)]

		train_set=pd.concat([train_set_1[:int(len(train_set_1))], train_set_0[:int(len(train_set_1))]])

		self.ids_train_split, self.ids_valid_split = train_test_split(train_set.index.values, test_size=0.2, random_state=42)


	def train_generator(self):
	    while True:
		shuffle(self.ids_train_split)
		for start in range(0, len(self.ids_train_split), self.batch_size):
		    x_batch = []
		    y_batch = []
		    end = min(start + self.batch_size, len(self.ids_train_split))
		    ids_train_batch = self.ids_train_split[start:end]
		    for id in ids_train_batch:
			row=self.df_train_scaled.loc[id]
					
			#row_scaled=np.append(row,[0])
			#y_append=np.append(self.df_labels.loc[id],[0])
			#row_scaled=np.append(row)
			#y_append=np.append(self.df_labels.loc[id])	
		        x_batch.append(row)
		        y_batch.append(self.df_labels.loc[id])

		    x_batch=np.array(x_batch)
		    y_batch=np.array(y_batch)
		    x_batch = np.reshape(x_batch, (x_batch.shape[0],x_batch.shape[1],1))
		    y_batch = np.reshape(y_batch, (y_batch.shape[0],y_batch.shape[1],1))
		    yield x_batch,y_batch



	def valid_generator(self):
	    #global batch_size, df_train_scaled, ids_train_split, ids_valid_split
	    while True:
		shuffle(self.ids_valid_split)
		for start in range(0, len(self.ids_valid_split), self.batch_size):
		    x_batch = []
		    y_batch = []
		    end = min(start + self.batch_size, len(self.ids_valid_split))
		    ids_valid_batch = self.ids_valid_split[start:end]
		    for id in ids_valid_batch:
			row=self.df_train_scaled.loc[id]
					
			#row_scaled=np.append(row,[0])
			#y_append=np.append(self.df_labels.loc[id],[0])
			#row_scaled=np.append(row)
			#y_append=np.append(self.df_labels.loc[id])	
		        x_batch.append(row)
		        y_batch.append(self.df_labels.loc[id])
		
		    x_batch=np.array(x_batch)
		    y_batch=np.array(y_batch)
		    x_batch = np.reshape(x_batch, (x_batch.shape[0],x_batch.shape[1],1))
		    y_batch = np.reshape(y_batch, (y_batch.shape[0],y_batch.shape[1],1))

		    yield x_batch,y_batch

