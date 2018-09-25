import csv
import numpy as np

SEX = ['male', 'female']
CABINS = []
EMBARKED = []

def prepare_train_data(data):
	try:
		new_data = []
		new_data.append(int(data[1])) # Survived
		new_data.append(float(data[2]) / 3.0) # Class
		new_data.append(SEX.index(data[4])) # Sex
		if (data[5] != ''):
			new_data.append(float(data[5]) / 100.0) # Age
		else:
			new_data.append(25)
		new_data.append(float(data[6]) / 10.0) # SibSp
		new_data.append(float(data[7]) / 10.0) # Parch
		new_data.append(float(data[9]) / 513.0) # Fare
		if not data[10] in CABINS:
			CABINS.append(data[10])
		new_data.append(CABINS.index(data[10])) # Cabin
		if not data[11] in EMBARKED:
			EMBARKED.append(data[11])
		new_data.append(EMBARKED.index(data[11])) # Embarked
		return new_data
	except:
		# print(data)
		pass

def prepare_test_data(data, survived):
	try:
		new_data = []
		new_data.append(float(data[1]) / 3.0) # Class
		new_data.append(SEX.index(data[3])) # Sex
		# if (data[4] != ''):
		# 	new_data.append(float(data[4]) / 100.0) # Age
		# else:
		# 	new_data.append(25)
		new_data.append(float(data[4]) / 100.0) # Age
		new_data.append(float(data[5]) / 10.0) # SibSp
		new_data.append(float(data[6]) / 10.0) # Parch
		new_data.append(float(data[8]) / 513.0) # Fare
		new_data.append(CABINS.index(data[9])) # Cabin
		new_data.append(EMBARKED.index(data[10])) # Embarked
		new_data.append(float(survived[1]))
		return new_data
	except:
		pass


f = open('data/train.csv', 'r')

data = []
r = csv.reader(f)
for row in r:
	data.append(prepare_train_data(row))

X = []
y = []
for d in data:
	if not d == None:	
		X.append(d[1:])
		y.append([d[0]])

# print(len(X))

X = np.array(X)
y = np.array(y)

f.close()

f1 = open('data/test.csv', 'r')
f2 = open('data/gender_submission.csv', 'r')

data = []
r1 = csv.reader(f1)
r2 = csv.reader(f2)
for row in r1:
	data.append(prepare_test_data(row, next(r2)))

X_val = []
y_val = []
for d in data:
	if not d == None:
		X_val.append(d[:-1])
		y_val.append([d[-1]])

X_val = np.array(X_val)
y_val = np.array(y_val)

# print(len(X_val))

f1.close()
f2.close()


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(100, activation = 'sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
model.fit(X, y, epochs = 100, batch_size = 8, validation_data = (X_val, y_val))
