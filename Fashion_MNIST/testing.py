from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras import backend as K
import numpy as np
import pandas as pd
import csv
import os
import theano

def backendSetup():
	K.set_image_dim_ordering('th'); print()
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	theano.config.optimizer = "None"
backendSetup()

x_test = [] #test data image info
w, h = 28, 28

y_answers = []

print('Reading CSV testing file...')
with open('input/test.csv', 'r') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
		if (row[0] != 'label'):  #check the csv file for row to skip
			pixels = np.array(row[1:]);
			img = np.reshape(pixels, (1, h, w))

			y_answers.append(row[0])

			x_test.append(img)
print('CSV testing file read.')

x_test = np.array(x_test)
y_answers = np.array(y_answers)

print('Loading model...')
json_file = open('ModelsAndFits/Models/Model3/Fit1/Model-3-1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('ModelsAndFits/Models/Model3/Fit1/Model-3-1.h5')
print('Model loaded.')

print('Testing model...')
finalGuesses = []
predictions = loaded_model.predict(x_test)
for i in range(0, len(x_test)):
	curr = 0
	high = 0
	while curr < len(predictions[i]):
		if predictions[i][curr] > predictions[i][high]:
			high = curr
		curr += 1
	finalGuesses.append(high)
print('Model tested.')

correct = 0
total = 0
print('Comparing to Answers...')
for i in range(0, len(y_answers)):
	if(int(y_answers[i]) == int(finalGuesses[i])):
		correct += 1
	total += 1
print("# Correct: " + str(correct))
print("# Total:   " + str(total))