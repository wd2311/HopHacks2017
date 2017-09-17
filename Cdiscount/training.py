# Learning gradient descent - Brain team at Google
# Network architecture search
# Fashion MNIST Github has other peoples' source code for ML models
# dropout 25-75%
# MaxPooling could also be different types of pooling e.g. Average (maybe Min)
# Jupiter notebooks - live coding interaction
# Saving and continuing to train

# Coursera - Andrew NG Deep Learning

# python profiling / performance tool
# python package: pickle - takes data and serializes it (turns it into a binary string)
# gridsearch cv - hasn't gotten better in 40 years - brute forces to find parameters

#keras add functionality to give you a 2 second interval to interupt to save the model

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

def readTrainingData():
	print("Reading training data...")
	data = bson.decode_file_iter(open('data/train.bson', 'rb'))

	prod_to_category = dict()
	count = 0

	x = []
	y = []

	for c, d in enumerate(data):
	    product_id = d['_id']
	    category_id = d['category_id'] # This won't be in Test data
	    prod_to_category[product_id] = category_id
	    for e, pic in enumerate(d['imgs']):
	        x.append(imread(io.BytesIO(pic['picture'])))
	        y.append(prod_to_category[product_id])

	def shrink(x, y, r): # This method randomly selects a percentage of the original data set as a sample
	    newX = []
	    newY = []

	    for i in range(0, int(len(x)*percenta)):
	        rand = random.randrange(len(x))
	        newX.append(x.pop(rand)) #pop removes and returns, avoids doubles
	        newY.append(y.pop(rand))

	    return (newX, newY)
	r = .2
	x, y = shrink(x, y, r)

	xTrain = np.array(x)
	y = np.array(y)

	uniques, id_train = np.unique(y, return_inverse=True)
	yTrain = np_utils.to_categorical(id_train, categories)

	print("Training data read.")

	return (xTrain, yTrain)

def backendSetup():
	K.set_image_dim_ordering('tf'); print()
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	theano.config.optimizer = "None" # delete this, see if still works

def createModel(shapeX):
	print('Creating model...')
	model = Sequential()

	#This layer will roughly turn 180x180 into 90x90
	model.add(Conv2D(8, (6, 6), strides=2, input_shape=shapeX, padding="same")) #Look into step size argument
	model.add(Activation('relu'));

	#This layer will roughly turn 90x90 into 45x45
	model.add(Conv2D(6, (4, 4), strides=2));
	model.add(Activation('relu'));

	#This layer will roughly turn 45x45 into 22.5x22.5
	model.add(MaxPooling2D(pool_size=(2, 2)));

	#This layer will roughly maintain size
	model.add(Conv2D(20, (2, 2), strides=1));
	model.add(Activation('relu'));

	#A 25% destroy-ratio dropout layer
	model.add(Dropout(0.25));

	#Flatten the 2D model into a 1D model
	model.add(Flatten());

	#A fully connected layer of 128 Nodes
	model.add(Dense(128));
	model.add(Activation('relu'));

	#A 25% destroy-ratio dropout layer
	model.add(Dropout(0.25));

	#A fully connected layer of 64 Nodes
	model.add(Dense(64));
	model.add(Activation('relu'));

	#A 25% destroy-ratio dropout layer
	model.add(Dropout(0.25));

	#A fully connected layer of *categories* nodes - so that all outputs are independent.
	model.add(Dense(categories));
	model.add(Activation('softmax')); #softmax means the output ratios will add to 1.

	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
	print('Model created.')
	return model

def trainModel(model, epochs, batchSize, xTrain, yTrain):
	nb_epoch = epochs;
	batch_size = batchSize; #data points * (epochs / batch_size) 
	model.fit(xTrain, yTrain, batch_size=batch_size, epochs=nb_epoch, verbose=1)
	print('A fitting stage has finished.')
	return model

def saveModel(model, modelNum, fitNum):
	print('Saving model...')
	model_json = model.to_json()
	filePath = 'ModelsAndFits/Models/Model' + modelNum + '/Fit' + fitNum + '/'
	fileName = 'Model-' + modelNum + '-' + fitNum;
	with open(filePath + fileName + ".json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights(filePath + fileName + ".h5")
	print('Model has been saved.')

def loadAndCompileModel(modelNum, fitNum): #
	print('Loading model...')
	filePath = 'ModelsAndFits/Models/Model' + modelNum + '/Fit' + fitNum + '/'
	fileName = 'Model-' + modelNum + '-' + fitNum;
	json_file = open(fileName + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(filePath + fileName + ".h5")
	print('Model loaded.')
	print('Compiling Model...')
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
	print('Model Compiled.')
	return loaded_model

backendSetup()

xTrain, yTrain = readTrainingData()

print(xTrain.shape); print()

model = createModel(xTrain.shape[1:])

model.summary();

def training(model):
	T = xTrain.shape[0]             # of trainingDataPoints
	C = categories                  # of categories

	B = 1                          # of batches
	M = .001                         # Approximately, What % of T should our first batch size be?     
	timeConstant = 1
	# I concluded that the amount of time you have to learn something affects the way you shhould learn it, this controls that
	# timeConstant has units "How many epochs should the first batch have"        
	#def SIC(percentOfStagesDone):  # "Stage Importance Weight"
		#return 1

	# I suspect - (B, k)(how much time we have)

	BS = []   # Batch Sizes
	E = []    # Epoch Numbers

	for i in np.arange(0.0 + (1/(2*B)), 1.0, (1/B)): # Start from half of your percentage increment, and then loop
		BS.append(int((T*M)/(pow((T*M)/C, i))))
		E.append(1/(T/BS[-1]))                    # Note: This is an intermediary step, these are arbitrary numbers forming non-arbitrary ratios for # epochs per stage

	E = np.array(E)
	scaleFactor = timeConstant/E[0]               # This step makes the factor that makes the numbers non-arbitrary
	E = (scaleFactor * E).astype(int)

	for stage in range(0, B):
		print("Stage: " + str(stage))
		print("  BatchSize: " + str(BS[stage]))
		print("  Epochs   : " + str(E[stage]))

	for stage in range(0, B):  # Batch Stage
	    model = trainModel(model, E[stage], BS[stage], xTrain, yTrain)
	    saveModel(model, '3', '1')

training(model)

# Parameter passing has been edited****: 
	# model = trainModel(model, 5, 60, x, y_train)
	# saveModel(model, 'Model')
	# loaded_model = loadAndCompileModel('Model')
	# loaded_model = trainModel(loaded_model, 10, 120, x, y_train)

# We're going to need to parallelize our model


# Tell you how long it is going to take to train a model given its parameters