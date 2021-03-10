
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import time

np.set_printoptions(threshold=sys.maxsize, precision = 3)

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ALGORITHM = "guesser"
#ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

DATASET = "mnist_d"
#DATASET = "mnist_f"
#DATASET = "cifar_10"
#DATASET = "cifar_100_f"
#DATASET = "cifar_100_c"

if DATASET == "mnist_d":
	NUM_CLASSES = 10
	IH = 28
	IW = 28
	IZ = 1
	IS = 784
elif DATASET == "mnist_f":
	NUM_CLASSES = 10
	IH = 28
	IW = 28
	IZ = 1
	IS = 784
elif DATASET == "cifar_10":
	NUM_CLASSES = 10
	IH = 32
	IW = 32
	IZ = 3
	IS = 1024
elif DATASET == "cifar_100_f":
	NUM_CLASSES = 100
	IH = 32
	IW = 32
	IZ = 3
	IS = 1024
elif DATASET == "cifar_100_c":
	NUM_CLASSES = 20
	IH = 32
	IW = 32
	IZ = 3
	IS = 1024

RANDOM_CROP_ENABLED = False
SAVE_LOAD_MODEL = True


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
	ans = []
	for entry in xTest:
		pred = [0] * NUM_CLASSES
		pred[random.randint(0, 9)] = 1
		ans.append(pred)
	return np.array(ans)


def buildTFNeuralNet(x, y, eps = 30):

	if SAVE_LOAD_MODEL:
		path = "Saved_Model_ANN_" + DATASET + "_Epochs_" + str(eps) + "_Random_Crop_" + str(RANDOM_CROP_ENABLED)
		if os.path.exists(path):
			return keras.models.load_model(path)
		else:
			print("No existing model found to load, training model")

	dropRate = 0.5
	categorical_crossentropy = keras.losses.CategoricalCrossentropy()
	net = keras.models.Sequential()
	net.add(tf.keras.layers.Dense(256, activation = tf.nn.sigmoid))
	net.add(tf.keras.layers.Dropout(dropRate))
	net.add(tf.keras.layers.Dense(256, activation = tf.nn.sigmoid))
	net.add(tf.keras.layers.Dropout(dropRate))
	net.add(tf.keras.layers.Dense(256, activation = tf.nn.sigmoid))
	net.add(tf.keras.layers.Dropout(dropRate))
	net.add(keras.layers.Dense(NUM_CLASSES, activation = tf.nn.sigmoid))
	net.compile(optimizer = 'adam', loss = categorical_crossentropy, metrics = [keras.metrics.CategoricalAccuracy()])
	net.fit(x, y, epochs = eps, verbose = 1)

	if SAVE_LOAD_MODEL:
		path = "Saved_Model_ANN_" + DATASET + "_Epochs_" + str(eps) + "_Random_Crop_" + str(RANDOM_CROP_ENABLED)
		net.save(path)

	return net


def buildTFConvNet(x, y, eps = 30, dropout = True, dropRate = 0.4):

	if SAVE_LOAD_MODEL:
		path = "Saved_Model_CNN_" + DATASET + "_Epochs_" + str(eps) + "_Random_Crop_" + str(RANDOM_CROP_ENABLED)
		if os.path.exists(path):
			return keras.models.load_model(path)
		else:
			print("No existing model found to load, training model")

	categorical_crossentropy = keras.losses.CategoricalCrossentropy()
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=9000, decay_rate = 0.98, staircase=True)
	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
	opt = keras.optimizers.Adam(learning_rate = lr_schedule)
	inShape = (IH, IW, IZ)

	net = keras.models.Sequential()

	if RANDOM_CROP_ENABLED:
		net.add(tf.keras.layers.experimental.preprocessing.RandomCrop(IH - 3, IW - 3))

	net.add(keras.layers.Conv2D(
		32,
		kernel_size=(3, 3),
		activation=tf.nn.relu,
		input_shape=inShape,
		padding='same',
		#kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
		#bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)
		))
	'''net.add(keras.layers.Conv2D(
		64,
		kernel_size=(3, 3),
		activation=tf.nn.relu,
		input_shape=inShape, padding='same',
		kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
		bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)
		))'''
	net.add(keras.layers.Conv2D(
		64,
		kernel_size=(3, 3),
		activation=tf.nn.relu,
		padding='same',
		#kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
		#bias_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)
		))
	net.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
	if dropout:
		net.add(tf.keras.layers.Dropout(dropRate))
	net.add(keras.layers.Flatten())
	net.add(keras.layers.BatchNormalization())
	net.add(keras.layers.Dense(512, activation=tf.nn.relu))
	if dropout:
		net.add(tf.keras.layers.Dropout(dropRate))
	net.add(keras.layers.Dense(256, activation=tf.nn.relu))
	if dropout:
		net.add(tf.keras.layers.Dropout(dropRate))
	net.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))

	opt = keras.optimizers.Adam(learning_rate = lr_schedule)
	#opt = keras.optimizers.Adam()

	net.compile(optimizer = opt, loss = categorical_crossentropy, metrics = [keras.metrics.CategoricalAccuracy()])
	net.fit(x, y, epochs = eps, verbose = 1, callbacks = [callback], batch_size=128, validation_split = 0.15)

	if SAVE_LOAD_MODEL:
		path = "Saved_Model_CNN_" + DATASET + "_Epochs_" + str(eps) + "_Random_Crop_" + str(RANDOM_CROP_ENABLED)
		net.save(path)

	return net


#=========================<Pipeline Functions>==================================

def getRawData():
	if DATASET == "mnist_d":
		mnist = tf.keras.datasets.mnist
		(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
	elif DATASET == "mnist_f":
		mnist = tf.keras.datasets.fashion_mnist
		(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
	elif DATASET == "cifar_10":
		cifar_10 = tf.keras.datasets.cifar10
		(xTrain, yTrain), (xTest, yTest) = cifar_10.load_data()
	elif DATASET == "cifar_100_f":
		cifar_100 = tf.keras.datasets.cifar100
		(xTrain, yTrain), (xTest, yTest) = cifar_100.load_data(label_mode="fine")
	elif DATASET == "cifar_100_c":
		cifar_100 = tf.keras.datasets.cifar100
		(xTrain, yTrain), (xTest, yTest) = cifar_100.load_data(label_mode="coarse")
	else:
		raise ValueError("Dataset not recognized.")
	print("Dataset: %s" % DATASET)
	print("Shape of xTrain dataset: %s." % str(xTrain.shape))
	print("Shape of yTrain dataset: %s." % str(yTrain.shape))
	print("Shape of xTest dataset: %s." % str(xTest.shape))
	print("Shape of yTest dataset: %s." % str(yTest.shape))
	return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
	((xTrain, yTrain), (xTest, yTest)) = raw
	xTrain = xTrain / 255
	xTest = xTest / 255
	if ALGORITHM != "tf_conv":
		# Multiply by IZ to account for number of channels
		xTrainP = xTrain.reshape((xTrain.shape[0], IS * IZ))
		xTestP = xTest.reshape((xTest.shape[0], IS * IZ))
	else:
		xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
		xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
	yTrainP = to_categorical(yTrain, NUM_CLASSES)
	yTestP = to_categorical(yTest, NUM_CLASSES)
	sh1 = xTrainP.shape
	sh2 = xTestP.shape
	sh3 = yTrainP.shape
	sh4 = yTestP.shape
	return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
	xTrain, yTrain = data
	if ALGORITHM == "guesser":
		return None   # Guesser has no model, as it is just guessing.
	elif ALGORITHM == "tf_net":
		print("Building and training TF_NN.")
		return buildTFNeuralNet(xTrain, yTrain)
	elif ALGORITHM == "tf_conv":
		print("Building and training TF_CNN.")
		return buildTFConvNet(xTrain, yTrain)
	else:
		raise ValueError("Algorithm not recognized.")



def runModel(data, model):
	if ALGORITHM == "guesser":
		return guesserClassifier(data)
	elif ALGORITHM == "tf_net":
		print("Testing TF_NN.")
		preds = model.predict(data)
		for i in range(preds.shape[0]):
			oneHot = [0] * NUM_CLASSES
			oneHot[np.argmax(preds[i])] = 1
			preds[i] = oneHot
		return preds
	elif ALGORITHM == "tf_conv":
		print("Testing TF_CNN.")
		preds = model.predict(data)
		for i in range(preds.shape[0]):
			oneHot = [0] * NUM_CLASSES
			oneHot[np.argmax(preds[i])] = 1
			preds[i] = oneHot
		return preds
	else:
		raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
	xTest, yTest = data
	acc = 0
	for i in range(preds.shape[0]):
		if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
	accuracy = acc / preds.shape[0]
	print("Classifier algorithm: %s" % ALGORITHM)
	print("Dataset: %s" % DATASET)
	print("Classifier accuracy: %f%%" % (accuracy * 100))
	print()
	if (NUM_CLASSES > 10):
		return

	# Actual values along y axis
	# Predicted values along x axis
	confusionMatrix = np.zeros([NUM_CLASSES, NUM_CLASSES])
	for i in range(preds.shape[0]):
		# Get the real output, as index in the confusion matrix
		labelIndex = np.argmax(yTest[i])

		# Get the predicted output, as index in the confusion matrix
		predictedIndex = np.argmax(preds[i])

		# Value is at index [label][predicted], because label along X and predicted along Y
		# We increment this value in the matrix
		confusionMatrix[labelIndex][predictedIndex] += 1

	print("Confusion Matrix:")
	print(confusionMatrix.astype(int))

	f1matrix = np.zeros([NUM_CLASSES])
	for i in range(NUM_CLASSES):
		tp = confusionMatrix[i][i]
		fp = 0	# Sums the column
		fn = 0	# Sums the row
		row = confusionMatrix[i]
		column = confusionMatrix[:, i]

		for j in range(NUM_CLASSES):
			if j != i:
				fp += column[j]
				fn += row[j]

		if (tp != 0 and fp != 0):
			precision = (float(0) + tp) / (float(0) + tp + fp)
		else:
			precision = float('inf')
		if (tp != 0 and fn != 0):
			recall = (float(0) + tp) / (float(0) + tp + fn)
		else:
			recall = float('inf')
		
		f1matrix[i] = (float(2) * precision * recall) / (float(0) + precision + recall)

	print("F1 Score Matrix")
	print(f1matrix)



#=========================<Main>================================================

def main():
	start = time.time()
	raw = getRawData()
	data = preprocessData(raw)
	model = trainModel(data[0])
	preds = runModel(data[1][0], model)
	evalResults(data[1], preds)
	print("\nTime Elapsed: %0.1f seconds" % (time.time() - start))

if __name__ == '__main__':
	main()
