import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics 
from sklearn.externals import joblib

if __name__ == '__main__':
	df = pd.read_csv('weibo.csv')
	x = df.iloc[:,1:]


# PCA降维
##计算全部贡献率
	n_components = 400
	x_pca = PCA(n_components = 100).fit_transform(x)


# SVM (RBF)
	print('SVM')
	clf = joblib.load("train_model_svm.m")
	pred_y = clf.predict(x_pca)
	df_pred_y = pd.DataFrame(pred_y)
	PRED = pd.concat([df_pred_y],axis = 1)
	PRED.to_csv('SVM_PRED.csv')
	print('Well Done!')

# BP
	print('BP')
	clf = joblib.load("train_model_BP.m")
	pred_y = clf.predict(x_pca)
	df_pred_y = pd.DataFrame(pred_y)
	PRED = pd.concat([df_pred_y],axis = 1)
	PRED.to_csv('BP_PRED.csv')
	print('Well Done!')

#randomforest
	print('RandomForest')
	clf = joblib.load("train_model_randomforest.m")
	pred_y = clf.predict(x_pca)
	df_pred_y = pd.DataFrame(pred_y)
	PRED = pd.concat([df_pred_y],axis = 1)
	PRED.to_csv('RandomForest_PRED.csv')
	print('Well Done!')


