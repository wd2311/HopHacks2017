from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "./data"]).decode("utf8"))
import io
import bson                       # this is installed with the pymongo package
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data
import random

categories = 5000 #5000 categories
w, h = 180, 180

def backendSetup():
	K.set_image_dim_ordering('th'); print()
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	theano.config.optimizer = "None"

def loadModel():
	print('Loading model...')
	json_file = open('ModelsAndFits/Models/Model1/Fit1/Model-1-1.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights('ModelsAndFits/Models/Model1/Fit1/Model-1-1.h5')
	print('Model loaded.')
	return loaded_model

def readTestingData():
	print('Reading testing data...')
	data = bson.decode_file_iter(open('data/test.bson', 'rb'))

	prod_to_category = dict()
	count = 0

	x1 = []
	x2 = []

	for c, d in enumerate(data):
	    product_id = d['_id']
	    for e, pic in enumerate(d['imgs']):
	        x1.append(imread(io.BytesIO(pic['picture'])))
	        x2.append(product_id)

	xTest = np.array(x1)
	IDs = np.array(x2)

	print('Testing data read.')

	return (xTest, IDs)

def testModel():
	print('Testing model...')
	finalGuesses = []
	predictions = loaded_model.predict(xTest)
	for i in range(0, len(xTest)):
		curr = 0
		high = 0
		while curr < len(predictions[i]):
			if predictions[i][curr] > predictions[i][high]:
				high = curr
			curr += 1
		finalGuesses.append(high)
	print('Model tested.')
	return finalGuesses

def writeCSVresults():
	print('Writing tested results to "resultsTest.csv"...')
	with open('resultsTest.csv', 'w', newline='') as csvfile:
	    writer = csv.writer(csvfile, delimiter=',')
	    writer.writerow(['_id', 'category_id'])
	    for i in range (0, len(finalGuesses)):
	    	writer.writerow([xIDs[i], finalGuesses[i]])
	print('Tested results written to "resultsTest.csv".')

backendSetup()
model = loadModel()
xTest, xIDs = readTestingData()
finalGuesses = testModel()
writeCSVresults()