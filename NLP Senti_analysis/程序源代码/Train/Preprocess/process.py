#训练集\验证集划分
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

if __name__ == '__main__':
	df = pd.read_csv('data.csv')
	y = df.iloc[:,1]
	x = df.iloc[:,2:]
	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.95)
	xtest = pd.DataFrame(x_test)
	ytest = pd.DataFrame(y_test)
	test = pd.concat([ytest,xtest],axis = 1)
	test.to_csv('test.csv')
	xtrain = pd.DataFrame(x_train)
	ytrain = pd.DataFrame(y_train)
	train = pd.concat([ytrain,xtrain],axis=1)
	train.to_csv('train.csv')




