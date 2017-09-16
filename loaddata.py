import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, scale, StandardScaler
from sklearn.utils import shuffle


class data:
	ev_label_1=[]
	ev_label_0=[]
	leftover_labels=[]
	#final scaled dataset, scaled by row on time series and scaled by column on mean,med,std
	df_binary_train_scaled_meanstd=pd.DataFrame()
	X_ids=[]
	
	def common_data_load(self):
		global ev_label_1, ev_label_0,df_binary_train_scaled_meanstd
		# load data

		df_train = pd.read_csv('EV_train.csv', index_col=0)

		df_train=df_train.T.fillna(value=0).T
		houses_w_nans=df_train[pd.isnull(df_train).any(axis=1)]
		houses_missing_data=houses_w_nans.index.tolist()
		df_train=df_train[~df_train.index.isin(houses_missing_data)]

		#Scale data in 0-1 range for neural network
		scaler = MinMaxScaler(feature_range=(0, 1))
		df_train_scaled = scaler.fit_transform(df_train.values.T).T
		df_train_scaled = pd.DataFrame(df_train_scaled, index=df_train.index, columns=df_train.columns)

		#labels data
		df_labels=pd.read_csv('EV_train_labels.csv', index_col=0)

		ev_label_all=df_labels.max(axis=1).rename('Has_EV')
		ev_label_1=ev_label_all.loc[ev_label_all.values==1]
		ev_label_0=ev_label_all.loc[ev_label_all.values==0]
		ev_label_all=pd.DataFrame(ev_label_all)
		ev_label_1=pd.DataFrame(ev_label_1)
		ev_label_0=pd.DataFrame(ev_label_0)

		#merge binary classification labels with scaled energy data
		df_binary_train=pd.merge(ev_label_all,df_train_scaled,left_index=True, right_index=True)

		#add features mean, std, median, scaled on the columns
		df_binary_train_unscaled=pd.merge(df_binary_train,df_train,left_index=True, right_index=True)
		df_binary_train_unscaled['mean'] = df_binary_train_unscaled[df_binary_train_unscaled.columns[2:]].values.mean(axis=1)
		df_binary_train_unscaled['std'] = df_binary_train_unscaled[df_binary_train_unscaled.columns[2:-1]].values.std(axis=1)
		df_binary_train_unscaled['median'] = np.median(df_binary_train_unscaled[df_binary_train_unscaled.columns[2:-2]].values,axis=1)

		#use standard scale across columns
		scaler = StandardScaler()
		df_train_scaled_meanstd = scaler.fit_transform(df_binary_train_unscaled[df_binary_train_unscaled.columns[-3:]].values)
		df_train_scaled_meanstd = pd.DataFrame(df_train_scaled_meanstd, index=df_binary_train_unscaled.index,columns=df_binary_train_unscaled.columns[-3:])


		#merge and shuffle all data
		df_binary_train_scaled_meanstd=pd.merge(df_binary_train,df_train_scaled_meanstd,left_index=True, right_index=True)
		df_binary_train_scaled_meanstd=shuffle(df_binary_train_scaled_meanstd)

			#return 

	def merge_pred_datasets(self,totals_best,totals_best2):
		
		final=pd.merge( totals_best[totals_best.columns[np.r_[0:11, 12]]],  totals_best2[totals_best2.columns[np.r_[1:11, 12]]],left_index=True, right_index=True)

		final['average']=final[final.columns[np.r_[1:11, 12:22]]].mean(axis=1)

		final['final_compare']=np.where((pd.to_numeric(final.final_bool_x) & pd.to_numeric(final.final_bool_y))==1, final.final_bool_x.astype('int64'),'0')# final['average']>0.35)

		condition = ((final['final_bool_x'].astype('int64') == 0) & (final['final_bool_y'].astype('int64')==0))	
		final['final_compare']=np.where(condition, '0' , final['final_compare'])

		condition = ((final['final_bool_x'].astype('int64') == 0) & (final['final_bool_y'].astype('int64')==1))	
		final['final_compare']=np.where(condition, '0' , final['final_compare'])

		condition = ((final['final_bool_x'].astype('int64') == 1) & (final['final_bool_y'].astype('int64')==0))	
		final['final_compare']=np.where(condition,(final['average']>0.15).astype('int64'), final['final_compare'].astype('int64'))

		final=final.loc[:, ['Y','final_bool_x','final_bool_y', 'average', 'final_compare']]
		return final
			


	def train_data_load_oversampled_0(self,X,y):
		unique, counts = np.unique(y, return_counts=True)

		percent=float(float(counts[1])/float(counts[0]))

		amount_to_add=int(counts[1]*percent-counts[0])
		und=df_binary_train_scaled_meanstd.loc[self.leftover_labels.index]
		und=shuffle(und)	
		#print und[und.columns[1:]].values[0:amount_to_add].shape
		X=np.concatenate((X,und[und.columns[1:]].values[0:amount_to_add]))
		y=np.concatenate((y,und[und.columns[0]].values[0:amount_to_add]))
		return X,y

	


	def train_data_load_oversampled_1(self, gen_test_data=True,proportion=1):
		global ev_label_1, ev_label_0,df_binary_train_scaled_meanstd, leftover_labels, X_ids


		#this is used for undersampling dataset 
		#this generates 1:1*proportion data   
		undersampled_labels=pd.concat([ev_label_1[:int(len(ev_label_1))], ev_label_0[:int(len(ev_label_1)*proportion)]])
		self.leftover_labels=pd.DataFrame(ev_label_0[int(len(ev_label_1)*proportion):])


		#filter undersampled data
		und=df_binary_train_scaled_meanstd.loc[undersampled_labels.index]
		und=shuffle(und)
		und=shuffle(und)

		#setup train and test sets, using Stratified Kfold for validation
		if gen_test_data is False:

			X=und[und.columns[1:]].values[0:int(len(und))]
			y=und[und.columns[0]].values[0:int(len(und))]

		else:
			'''X=und[und.columns[1:]].values[np.r_[0:int(int(len(und)/2)*00.85),int(len(und)/2):int(len(und)/2)+int(int(len(und)/2)*00.85)]]
			y=und[und.columns[0]]. values[np.r_[0:int(int(len(und)/2)*00.85),int(len(und)/2):int(len(und)/2)+int(int(len(und)/2)*00.85)]]
			
			Xtest=und[und.columns[1:]].values[np.r_[int(int(len(und)/2)*00.85):int(len(und)/2):,int(len(und)/2)+int(int(len(und)/2)*00.85):len(und)]]	
			Ytest=und[und.columns[0]].values[np.r_[int(int(len(und)/2)*00.85):int(len(und)/2):,int(len(und)/2)+int(int(len(und)/2)*00.85):len(und)]]
			Xtest_ids=und.index.values[np.r_[int(int(len(und)/2)*00.85):int(len(und)/2):,int(len(und)/2)+int(int(len(und)/2)*00.85):len(und)]]'''
			X=und[und.columns[1:]].values[0:int(len(und)*00.85)]
			self.X_ids=und.index.values[0:int(len(und)*00.85)]

			y=und[und.columns[0]]. values[0:int(len(und)*00.85)]
			
			Xtest=und[und.columns[1:]].values[int(len(und)*00.85):]	
			Ytest=und[und.columns[0]].values[int(len(und)*00.85):]
			Xtest_ids=und.index.values[int(len(und)*00.85):]
			#set test dataset equivalent proportions to real dataset (300.85% 1's)
			#print "Proportion of 1's (Has_EV) in dataset: " +str(float(len(ev_label_1))/float((len(ev_label_1)+len(ev_label_0))))
			#algebra!

			unique, counts = np.unique(Ytest, return_counts=True)		
			#print counts
		
			ev_0_prop=float(len(ev_label_0))/float((len(ev_label_1)+len(ev_label_0)))
			ev_1_prop=float(len(ev_label_1))/float((len(ev_label_1)+len(ev_label_0)))
			No_EV_to_add=int((int(-ev_1_prop*counts[0])+int(ev_0_prop*counts[1]))/0.3)

			self.leftover_labels=shuffle(self.leftover_labels)
			newTestDataframe=pd.DataFrame(df_binary_train_scaled_meanstd.loc[self.leftover_labels.index[0:No_EV_to_add]].values, index=self.leftover_labels.index[0:No_EV_to_add])

			self.leftover_labels=self.leftover_labels[No_EV_to_add:]
			newX=newTestDataframe[newTestDataframe.columns[1:]].values
			newY=newTestDataframe[newTestDataframe.columns[0]].values
			newX_ids=newTestDataframe.index.values

			Xtest=np.append(Xtest, newX, axis=0)
			Ytest=np.append(Ytest, newY, axis=0)
			Xtest_ids=np.append(Xtest_ids,newX_ids, axis=0)


			totals = pd.DataFrame() 
			totals['Y'] = pd.Series(Ytest, index=Xtest_ids)

			totals_x=pd.DataFrame(Xtest, index=Xtest_ids)
			#end set test dataset equivalent proportions to real dataset
		return X, y, totals, totals_x



		
		



