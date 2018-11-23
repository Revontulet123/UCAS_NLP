import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC 
from sklearn.externals import joblib

if __name__ == '__main__':
	df = pd.read_csv('train.csv')
	y = df.iloc[:,1]
	x = df.iloc[:,2:]


# PCA降维
##计算全部贡献率
	n_components = 400
	x_pca = PCA(n_components = 100).fit_transform(x)


	clf = RFC(n_estimators=50)
	clf.fit(x_pca,y)
	joblib.dump(clf, "...\\train_model_randomforest.m")#模型存储路径
	print('Well Done.')



