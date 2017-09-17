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

categories = 10 #10 digits
w, h = 28, 28

def backendSetup():
	K.set_image_dim_ordering('th'); print()
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	theano.config.optimizer = "None" # delete this, see if still works

def readTrain():
	print('Reading CSV training file...')
	with open('input/train.csv', 'r') as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		x = []
		y = []
		for row in readCSV:
			if (row[0] != 'label'):
				pixels = np.array(row[1:785]);
				img = np.reshape(pixels, (1, h, w))

				#This is extremely stupid:
				# data = np.zeros((h, w, 3), dtype=np.uint8)
				# for i in range (0, (h*w)):
				# 	xPos = int(i/w)
				# 	yPos = i - w*int(i/w)
				# 	pixVal = int(pixels[i])
				# 	data[xPos, yPos] = [pixVal, pixVal, pixVal]
				# img = Image.fromarray(data, 'RGB')
				# img = img_to_array(img) / 255
				# img = img.transpose(2, 0, 1)
				# img = img.reshape(3, h, w)

				x.append(img)
				y.append(int(row[0]))
	x = np.array(x)
	y = np.array(y)
	print('CSV training file read.')

	uniques, id_train = np.unique(y, return_inverse=True)
	y_train = np_utils.to_categorical(id_train, categories)

	return (x, y_train)

def createModel(shapeX):
	print('Creating model...')
	model = Sequential()

	nb_filters1 = 6
	nb_convX1 = 3
	nb_convY1 = 3
	model.add(Conv2D(nb_filters1, (nb_convX1, nb_convY1), input_shape=shapeX, padding="same")) #Look into step size argument
	model.add(Activation('relu'));

	nb_filters2 = 4
	nb_convX2 = 3
	nb_convY2 = 3
	model.add(Conv2D(nb_filters2, (nb_convX2, nb_convY2)));
	model.add(Activation('relu'));

	nb_poolX1 = 2
	nb_poolY1 = 2
	model.add(MaxPooling2D(pool_size=(nb_poolX1, nb_poolY1)));

	nb_filters3 = 3
	nb_convX3 = 2
	nb_convY3 = 2
	model.add(Conv2D(nb_filters3, (nb_convX3, nb_convY3)));
	model.add(Activation('relu'));

	nb_poolX1 = 2
	nb_poolY1 = 2
	model.add(MaxPooling2D(pool_size=(nb_poolX1, nb_poolY1)));

	dropRatio1 = 0.5
	model.add(Dropout(dropRatio1));

	model.add(Flatten());

	denseNodes1 = 64
	model.add(Dense(denseNodes1)); # 64 then 32 from 28
	model.add(Activation('relu'));

	dropRatio2 = 0.5
	model.add(Dropout(dropRatio2));


	denseNodes2 = 32
	model.add(Dense(denseNodes2));
	model.add(Activation('relu'));

	dropRatio3 = 0.5
	model.add(Dropout(dropRatio3));


	model.add(Dense(categories));
	model.add(Activation('softmax')); #softmax means add to 1, 


	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
	print('Model created.')
	return model

def trainModel(model, epochs, batchSize, xTrain, yTrain):

	nb_epoch = epochs;
	batch_size = batchSize; #data points * (epochs / batch_size) 

	model.fit(xTrain, yTrain, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
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

xTrain, yTrain = readTrain() #training data image info, corresponding labels
model = createModel(xTrain.shape[1:])

model.summary();

T = xTrain.shape[0]             # of trainingDataPoints
C = categories                  # of categories

B = 6                          # of batches
M = .002                         # Approximately, What % of T should our first batch size be?     
timeConstant = 20
# I concluded that the amount of time you have to learn something affects the way you shhould learn it, this controls that
# timeConstant has units "How many epochs should the first batch have"        
#def SIC(percentOfStagesDone):  # "Stage Importance Weight"
	#return 1

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

# Parameter passing has been edited****: 
	# model = trainModel(model, 5, 60, x, y_train)
	# saveModel(model, 'Model')
	# loaded_model = loadAndCompileModel('Model')
	# loaded_model = trainModel(loaded_model, 10, 120, x, y_train)

# We're going to need to parallelize our model


# Tell you how long it is going to take to train a model given its parameters