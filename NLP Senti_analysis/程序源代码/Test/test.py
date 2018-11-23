import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics 
from sklearn.externals import joblib

if __name__ == '__main__':
	df = pd.read_csv('test.csv')
	y = df.iloc[1:,1]
	x = df.iloc[1:,2:]


# PCA降维
##计算全部贡献率
	n_components = 400
	x_pca = PCA(n_components = 100).fit_transform(x)


# SVM (RBF)

	clf = joblib.load("train_model_svm.m")
	pred_y = clf.predict(x_pca)
	print('SVM Test Accuracy: %.2f'% clf.score(x_pca,y))
	print(metrics.classification_report(y, pred_y))

# BP
	clf = joblib.load("train_model_BP.m")
	pred_y = clf.predict(x_pca)
	print('BP Test Accuracy: %.2f'% clf.score(x_pca,y))
	print(metrics.classification_report(y, pred_y))

#randomforest
	clf = joblib.load("train_model_randomforest.m")
	pred_y = clf.predict(x_pca)
	print('RandomForset Test Accuracy: %.2f'% clf.score(x_pca,y))
	print(metrics.classification_report(y, pred_y))


