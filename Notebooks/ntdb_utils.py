import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from functools import reduce
import shap
import matplotlib.pyplot as plt

def train_test_creator_CI(input_df,training_percent,column_to_divide): 
    #Divide into deceased and survived dfs
    zero_df = input_df[input_df[column_to_divide] < 0.5]
    one_df = input_df[input_df[column_to_divide] > 0.5]
    
    unique_inc_key = zero_df['INC_KEY'].unique()
    np.random.shuffle(unique_inc_key)    
    key_cut = int(unique_inc_key.size*training_percent) 
    zero_train = unique_inc_key[0:key_cut]
    zero_test = unique_inc_key[key_cut:unique_inc_key.size]  
    
    zero_training_keys_df = pd.DataFrame(data=zero_train)
    zero_testing_keys_df = pd.DataFrame(data=zero_test)
    zero_training_keys_df.columns = ['INC_KEY']
    zero_testing_keys_df.columns = ['INC_KEY']    
 
    zero_train_df = zero_training_keys_df.merge(input_df,left_on='INC_KEY', right_on='INC_KEY', how='inner')
    zero_test_df = zero_testing_keys_df.merge(input_df,left_on='INC_KEY', right_on='INC_KEY', how='inner')
    
    unique_inc_key = one_df['INC_KEY'].unique()
    np.random.shuffle(unique_inc_key)
    one_train = unique_inc_key[0:key_cut]
    one_test = unique_inc_key[key_cut:unique_inc_key.size] 
    
    one_training_keys_df = pd.DataFrame(data=one_train)
    one_testing_keys_df = pd.DataFrame(data=one_test)
    one_training_keys_df.columns = ['INC_KEY']
    one_testing_keys_df.columns = ['INC_KEY']  
    
    one_train_df = one_training_keys_df.merge(input_df,left_on='INC_KEY', right_on='INC_KEY', how='inner')
    one_test_df = one_testing_keys_df.merge(input_df,left_on='INC_KEY', right_on='INC_KEY', how='inner')
    
    #Combine balanced survived and deceased into training and test sets.
    train_df = pd.concat([zero_train_df,one_train_df])
    test_df = pd.concat([zero_test_df,one_test_df])
    
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    
    return train_df,test_df

def mutually_exclusive_df(inc_keys,column): 
	#column : column of interest, comorbidity column for instance. 
	#inc_keys: incident keys to identify patients
	return pd.crosstab(inc_keys, column,dropna=True)

def merge_mutliple_dfs(df_list,key,how): 
	return reduce(lambda  left,right: pd.merge(left,right,on=key,how=how), df_list)

def filterRealVals(df,prop_list): 
	for prop in prop_list: 
		df = df[df[prop] > 0]
	return df

def removeUnknownOutcomes(df,col_name,rem_list): 
	for rem in rem_list: 
		df = df[df[col_name] != rem]
	return df

def construct_feature_list(columns,rem_list): 
	feature_list = list(columns)
	for rem in rem_list: 
		feature_list.remove(rem)
	return feature_list

def getMissingData(all_data_df,completed_data): 
	common = all_data_df.merge(completed_data,on=['INC_KEY','INC_KEY'])
	return all_data_df[(~all_data_df.INC_KEY.isin(common.INC_KEY))]

def printMissingBreakdown(df,features): 
	for feat in features: 
		no = df[feat].isnull().sum()
		print(feat + ' is missing ' + str(no) + ' out of its ' + str(df.shape[0])+ ' entries')

def construct_missing_flags(df):
	missing_flags = df.isnull()*1
	missing_flags.columns = [str(col) + '_missing_flag' for col in missing_flags.columns]
	missing_flags["sum"] = missing_flags.sum(axis=1)
	return missing_flags

def excludedDataStats(df,flags,cutoff): 
	t = flags[flags['sum'] > cutoff].reset_index()
	missing_data_df_temp = df.reset_index()
	missing_data_df_temp = missing_data_df_temp.merge(t,left_on='index',right_on='index',how='inner')
	deathrate = 1 - missing_data_df_temp.OUTCOME.sum()/missing_data_df_temp.shape[0]
	print('Death rate of exluded data is ' + str(deathrate))

def dataToImpute(df,flags,features): 
	df = df.reset_index()
	flags = flags.reset_index()
	df = df.merge(flags,left_on='index',right_on='index',how='inner')
	return df[features]

def hist(df,features):
	for feat in features: 
		df[feat].hist()
		print(df[feat].describe())
		plt.title(feat)
		plt.show()


def constructSets(train,test,feature_list,outcome): 
	X_train = train[feature_list]
	X_test = test[feature_list]
	y_train = train[[outcome]]
	y_test = test[[outcome]]
	return X_train,X_test,y_train,y_test


def combineSets(x1,x2,y1,y2,flag): 
	if flag == 1: 
		xtot = pd.concat([x1,x2])
		ytot = pd.concat([y1,y2])
	return xtot,ytot

def encode(df,props): 
	lb_make = LabelEncoder()
	if 'GENDER' in props: 
		df["GENDER"] = lb_make.fit_transform(df["GENDER"])
	df["HOSPDISP_CODE"] = lb_make.fit_transform(df["HOSPDISP"])
	df['OUTCOME'] = df['HOSPDISP_CODE'].apply(lambda x: 1 if x>0 else 0)
	return df

def get_auc(X,y,clf,set_choice): 
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)[::,1]

    fpr, tpr, _ = metrics.roc_curve(y.values,  y_pred_proba)
    auc_test = metrics.roc_auc_score(y.values, y_pred_proba)

    label = str(set_choice) + ' AUC : ' + str(auc_test)
    print(label)
    return y_pred,y_pred_proba

def get_predictions(labels,scaler,clf,X,feature_list): 
	a = labels.values.ravel() == clf.predict(scaler.transform(X.values.reshape(-1,len(feature_list))))
	X = pd.DataFrame(scaler.inverse_transform(X.values),columns = feature_list)
	X.columns = ['Age','GCSTOT','HR','SBP','Temp','Gender', 'RR','SaO2']
	mistakes = [i for i, x in enumerate(a) if ~x]
	correct = [i for i, x in enumerate(a) if x]

	return mistakes,correct,X

def forceplot(shap,X,labels,y_pred,explainer,shap_values,group): 
	t = group
	ran_example = np.random.choice(len(t), 1, replace=False)[0]
	ran_example = t[ran_example]
	print('sample number is ' + str(ran_example))

	outc = 'deceased'
	if labels.iloc[ran_example][0] == 1:
		outc = 'survived'
	print('True outcome is ' + outc)

	outc = 'deceased'
	if y_pred[ran_example] == 1: 
		outc = 'survived'
	print('Predicted outcome is ' + outc)


	plt.figure()
	shap.force_plot(explainer.expected_value, shap_values[ran_example,:], X.iloc[ran_example,:],plot_cmap=["#0000ff","#FF0000"])










