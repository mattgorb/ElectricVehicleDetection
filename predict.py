import numpy as np
import os
import sys
import pandas as pd
import csv

def predict_final(model, final, batch_size, df_test):
	model.load_weights("unet_weights/weights.h5")
	preds_all=[]
	y_all=[]
	lines=[]
	predicted_1s=[]

	for i in range(len(final.index.values)):
		if(final.iloc[i]['average']<.2):
			row=np.insert(np.zeros((1,2880)), 0,final.index.values[i])
			lines.append(row)
		else:
			predicted_1s.append(final.index.values[i])

	for start in range(0, len(predicted_1s), batch_size):
	    x_batch = []
	    y_batch = []
	    end = min(start + batch_size, len(predicted_1s))
	    batch = predicted_1s[start:end]
	    for id in batch:
		row=df_test.loc[id]	
		x_batch.append(row)		

	    x_batch=np.array(x_batch)
	    x_batch=np.reshape(x_batch, (x_batch.shape[0],x_batch.shape[1],1))
	    preds=model.predict_on_batch(x_batch)
	    preds=np.reshape(preds, (preds.shape[0],preds.shape[1]))
	    tol = 0.005
	    preds.real[abs(preds.real) < tol] = 0
	    preds=np.array(preds, dtype='float')     
	    preds_all.extend(preds)

	preds_all=np.array(preds_all)
	preds_all= preds_all[:, :-1]

	for x in range(len(predicted_1s)):
		row2=np.insert(np.array(preds_all[x]), 0,predicted_1s[x])
		lines.append(np.array(row2))
	
	lines=np.array(lines)
	with open("test_results.csv", "wb") as f:
		writer=csv.writer(f, delimiter=',')
		writer.writerows(lines)







def predict(model, number_right, threshold, totals, totals_x, batch_size, path, ones_upsampled=True):
	try:
		#print "Threshold=" +str(threshold) + " with correct K folds=" +str(number_right)+"\n"

		for i in os.listdir(path):
			model.load_weights(path+i)
			preds_all=[]

			for start in range(0, len(totals_x), batch_size):
				end = min(start + batch_size, len(totals_x))
				Xtestbatch = totals_x.iloc[start:end].values
				preds=model.predict_on_batch(Xtestbatch)[:,0].tolist()
				preds_all.extend(preds)
			p=pd.DataFrame(preds_all, columns=['Predictions_'+str(i)],index=totals_x.index)

			totals=pd.merge(totals,p,left_index=True, right_index=True)


		totals['sum_gt_threshold'] = (totals[totals.columns[1:]]>threshold).sum(axis=1)
		totals['final_bool'] = np.where(totals['sum_gt_threshold']>=number_right, '1', '0')	
		totals['final_compare']=pd.to_numeric(totals.final_bool)==pd.to_numeric(totals.Y)
		#confusion matrix
		confusion_matrix= pd.crosstab(totals['Y'].values, totals['final_bool'].values, rownames=['Actual EV'], colnames=['Predicted EV'])
		#print confusion_matrix
		
		#print totals['final_compare'].value_counts()
		if(ones_upsampled):
			EV_correct=float(confusion_matrix.loc[1.0][1]/float(confusion_matrix.loc[1.0][1]+confusion_matrix.loc[1.0][0]))
		else:
			EV_correct=float(confusion_matrix.loc[.0][1]/float(confusion_matrix.loc[1.0][1]+confusion_matrix.loc[1.0][0]))
		#print "Predicted Has_EV correctly " +str(EV_correct)
		percent_correct=float(float(totals['final_compare'].value_counts()[1])/float(totals['final_compare'].value_counts()[0]+totals['final_compare'].value_counts()[1]))
		#print "Total percent correct: " +str(percent_correct)+"\n\n\n"
		totals_filtered=pd.merge(pd.DataFrame(totals.loc[totals['final_bool']=='1','Y']),totals_x,left_index=True, right_index=True)
		return EV_correct, totals
	except:
		print("Unexpected error:", sys.exc_info()[0])


def get_best_threshold(totals, totals_x, model, path, batch_size):
	num_right=[2,3,4,5,6,7,8,9]
	num_right_best=0
	threshold=[.25,.3,.4,.5,.6,.7,.8,.9]
	threshold_best=0
	max_EV_correct=0
	totals_best=0
	for i in num_right:
		for j in threshold:
			try:
				Has_EV_correct, predicted_EV=predict(model, i,j, totals, totals_x,batch_size, path)	
				if(float(Has_EV_correct)>=float(max_EV_correct)):
					max_EV_correct=Has_EV_correct
					num_right_best=i
					threshold_best=j
					totals_best=predicted_EV
			except: 
				print("Unexpected error:", sys.exc_info()[0])
	print "Best number = " +str(num_right_best)
	print "Best threshold = "+str(threshold_best)

	return totals_best, num_right_best,threshold_best




def predict_on_test(model, number_right, threshold, totals_x, batch_size, path):
	#try:
		#print "Threshold=" +str(threshold) + " with correct K folds=" +str(number_right)+"\n"
		totals=None
		for i in os.listdir(path):
			model.load_weights(path+i)
			preds_all=[]

			for start in range(0, len(totals_x), batch_size):
				end = min(start + batch_size, len(totals_x))
				Xtestbatch = totals_x.iloc[start:end].values
				preds=model.predict_on_batch(Xtestbatch)[:,0].tolist()
				preds_all.extend(preds)
			p=pd.DataFrame(preds_all, columns=['Predictions_'+str(i)],index=totals_x.index)
			
			if(totals is None):
				totals=p
			else:
				totals=pd.merge(totals,p,left_index=True, right_index=True)


		totals['sum_gt_threshold'] = (totals[totals.columns[1:]]>threshold).sum(axis=1)
		totals['final_bool'] = np.where(totals['sum_gt_threshold']>=number_right, '1', '0')	
		#totals['final_compare']=pd.to_numeric(totals.final_bool)==pd.to_numeric(totals.Y)

		#totals_filtered=pd.merge(pd.DataFrame(totals.loc[totals['final_bool']=='1','Y']),totals_x,left_index=True, right_index=True)
		print totals
		return totals

