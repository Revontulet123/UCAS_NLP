import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier 
from sklearn.externals import joblib

if __name__ == '__main__':
	df = pd.read_csv('train.csv')
	y = df.iloc[:,1]
	x = df.iloc[:,2:]


# PCA降维
##计算全部贡献率
	n_components = 400
	x_pca = PCA(n_components = 100).fit_transform(x)


	clf = MLPClassifier(solver='adam', alpha=1e-6,hidden_layer_sizes=(6,3,2), max_iter = 100000, random_state=1)
	clf.fit(x_pca,y)
	joblib.dump(clf, "...\\train_model_BP.m")#模型存储路径
	print('Well Done.')


