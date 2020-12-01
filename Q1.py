import numpy as np 
import pandas as pd

class MyNeuralNetwork():
	"""
	My implementation of a Neural Network Classifier.
	"""

	acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
	weight_inits = ['zero', 'random', 'normal']

	def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
		"""
		Initializing a new MyNeuralNetwork object

		Parameters
		----------
		n_layers : int value specifying the number of layers

		layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

		activation : string specifying the activation function to be used
					 possible inputs: relu, sigmoid, linear, tanh, softmax

		learning_rate : float value specifying the learning rate to be used

		weight_init : string specifying the weight initialization function to be used
					  possible inputs: zero, random, normal

		batch_size : int value specifying the batch size to be used

		num_epochs : int value specifying the number of epochs to be used
		"""

		if activation not in self.acti_fns:
			raise Exception('Incorrect Activation Function')

		if weight_init not in self.weight_inits:
			raise Exception('Incorrect Weight Initialization Function')

		ind = self.weight_inits.index(weight_init)

		if(ind == 0) :
			w_init = self.zero_init
		elif(ind == 1) :
			w_init = self.random_init
		else :
			w_init = self.normal_init

		self.W = []	# weights
		self.X = []	# value of perceptron
		self.B = []	# bias associated with perceptron

		# FIXME : It would be greate to complete the list into numpy array

		for i in range(1, n_layers) :
			self.W.append(w_init((layer_sizes[i-1] , layer_sizes[i])))
			self.X.append(w_init((layer_sizes[i] , )))				# just filling the space. however it would update in forward and backward phase, and does't affect algo based on weight_inits
			self.B.append(w_init((layer_sizes[i] , )))

		# print(W[1] , W[1][0] , W[1][0][0])
		ind = self.acti_fns.index(activation)

		if(ind == 0) :
			self.activation = self.relu
		elif(ind == 1) :
			self.activation = self.sigmoid
		elif(ind == 2) :
			self.activation = self.linear
		elif(ind == 3) :
			self.activation = self.tanh
		else :
			self.activation = self.softmax

		self.n_layers = n_layers
		self.layer_sizes = layer_sizes
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.num_epochs = num_epochs

	def relu(self, X):
		"""
		Calculating the ReLU activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return np.maximum(0,X)

	def relu_grad(self, X):
		"""
		Calculating the gradient of ReLU activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		X[X<=0] = 0
		X[X>0] = 1
		return X

	def sigmoid(self, X):
		"""
		Calculating the Sigmoid activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return 1 / (1+ np.exp(-X))

	def sigmoid_grad(self, X):
		"""
		Calculating the gradient of Sigmoid activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		sig = self.sigmoid(z)
		return sig * (1-sig)

	def linear(self, X):
		"""
		Calculating the Linear activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return X

	def linear_grad(self, X):
		"""
		Calculating the gradient of Linear activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return 1

	def tanh(self, X):
		"""
		Calculating the Tanh activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return np.tanh(X)

	def tanh_grad(self, X):
		"""
		Calculating the gradient of Tanh activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return 1 - np.power(np.tanh(X),2)

	def softmax(self, X):
		"""
		Calculating the ReLU activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		numerator = np.exp(X)
		return numerator / np.sum(numerator)

	def softmax_grad(self, X):
		"""
		Calculating the gradient of Softmax activation for a particular layer

		Parameters
		----------
		X : 1-dimentional numpy array 

		Returns
		-------
		x_calc : 1-dimensional numpy array after calculating the necessary function over X
		"""
		return None # TODO : need to complete softmax_grad

	def zero_init(self, shape):
		"""
		Calculating the initial weights after Zero Activation for a particular layer

		Parameters
		----------
		shape : tuple specifying the shape of the layer for which weights have to be generated 

		Returns
		-------
		weight : 2-dimensional numpy array which contains the initial weights for the requested layer
		"""
		#  [np.random.randn(y,1) for y in a[:]]		for bias

		return np.zeros(shape)

	def random_init(self, shape):
		"""
		Calculating the initial weights after Random Activation for a particular layer

		Parameters
		----------
		shape : tuple specifying the shape of the layer for which weights have to be generated 

		Returns
		-------
		weight : 2-dimensional numpy array which contains the initial weights for the requested layer
		"""
		return np.random.randn(shape[0] , shape[1])			# FIXME : what is diffrece between rand and randn

	def normal_init(self, shape):
		"""
		Calculating the initial weights after Normal(0,1) Activation for a particular layer

		Parameters
		----------
		shape : tuple specifying the shape of the layer for which weights have to be generated 

		Returns
		-------
		weight : 2-dimensional numpy array which contains the initial weights for the requested layer
		"""
		return None # TODO : implement this

	def forward_phase(self , input_layer):
		# for taking the inputs from input layer, hence appending input layer at index = -1
		self.X.append(input_layer)

		for i in range(1, self.n_layers) : # traversing from 1 to n_layer because layer 1 will use its synaptic weight (b/w layer 0 and 1) and layer 1 will be updated based on bias of layer
			for j in range(self.layer_sizes[i]) :	# traversing over perceptron in specified layer
				self.X[i-1][j] = np.dot(self.W[i-1][:,j] , self.X[i-2]) + self.B[i-1][j]			# storing the z function, activation function would be aplide over whole layer
				# here X[i-1] is denoting perceptron of current layer, however X[i-2] denotes perceptron of previous layer
				# W[i-1] denotes all the synaptic weights between layer i-1 and i.
				# Similarly, B[i-1] is bias associated with current layer
			self.X[i-1] = self.activation(self.X[i-1])				# appling activation after successfully calucalting z function for all perceptron in layer
		self.X.pop()

	def fit(self, X, y):
		"""
		Fitting (training) the linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

		y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
		
		Returns
		-------
		self : an instance of self
		"""

		# fit function has to return an instance of itself or else it won't work with test.py
		# print(self.activation , self.batch_size , self.num_epochs)

		"""
		TODO : what to implement in fit function

		1. feed the dataset X , y to network
		2. compute cost at output layer
		3. based upon previous step, do back_propogation
		"""

		self.W[0][0][0] = 1
		self.W[1][0][0] = 1

		# forward phase
		self.forward_phase(np.array([1.0,2.0,3.0]))

		print('----')

		print(self.X)

		return self

	def predict_proba(self, X):
		"""
		Predicting probabilities using the trained linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

		Returns
		-------
		y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """

		# return the numpy array y which contains the predicted values
		return None

	def predict(self, X):
		"""
		Predicting values using the trained linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

		Returns
		-------
		y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
		"""

		# return the numpy array y which contains the predicted values
		return None

	def score(self, X, y):
		"""
		Predicting values using the trained linear model.

		Parameters
		----------
		X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

		y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

		Returns
		-------
		acc : float value specifying the accuracy of the model on the provided testing set
		"""

		# return the numpy array y which contains the predicted values
		return None

# MyNeuralNetwork(4,[784,500,200,10] , 'sigmoid' , 0.01 , 'zero' , 100 ,  100)
# a = MyNeuralNetwork(3 , [3,2,1] , 'sigmoid' , 0.01 , 'zero' , 100 ,  100)
# a.fit([1],[1])
# MyNeuralNetwork(3 , [3,2,1] , 'sigmoid' , 0.01 , 'random' , 100 ,  100)

a = MyNeuralNetwork(3 , [3,2,1] , 'tanh' , 0.01 , 'zero' , 100 ,  100)
a.fit([1],[1])