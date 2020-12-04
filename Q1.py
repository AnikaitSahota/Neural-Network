import numpy as np 
# import pandas as pd

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
		self.B = []	# bias associated with perceptron

		# FIXME : It would be greate to complete the list into numpy array

		for i in range(1, n_layers) :
			self.W.append(w_init((layer_sizes[i-1] , layer_sizes[i])))
			self.B.append(w_init((layer_sizes[i] , )))

		# print(W[1] , W[1][0] , W[1][0][0])
		ind = self.acti_fns.index(activation)

		if(ind == 0) :
			self.activation = self.relu
			self.activation_derivative = self.relu_grad
		elif(ind == 1) :
			self.activation = self.sigmoid
			self.activation_derivative = self.sigmoid_grad
		elif(ind == 2) :
			self.activation = self.linear
			self.activation_derivative = self.linear_grad
		elif(ind == 3) :
			self.activation = self.tanh
			self.activation_derivative = self.tanh_grad
		else :
			self.activation = self.softmax
			self.activation_derivative = self.sigmoid_grad

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
		return X * (1-X)

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



	def __forward_phase(self , input_layer):
		"""
		Feed the input layer to network, activate the output layer perceptron

		Parameters
		----------
		input_layer : 1-dimensional numpy array which contains actication energy of input layer perceptron
		"""
		# for taking the inputs from input layer, hence appending input layer at index = -1
		A = [a.copy() for a in self.B]			# cloning B for getting accurate shapes of layers
		A.append(input_layer)					# we could have used insert(0,input_layer) but append is time efficient w.r.t new mempry allocation.

		for i in range(1, self.n_layers) : # traversing from 1 to n_layer because layer 1 will use its synaptic weight (b/w layer 0 and 1) and layer 1 will be updated based on bias of layer
			for j in range(self.layer_sizes[i]) :	# traversing over perceptron in specified layer
				A[i-1][j] = np.dot(self.W[i-1][:,j] , A[i-2]) + self.B[i-1][j]			# storing the z function, activation function would be aplide over whole layer
				# here A[i-1] is denoting perceptron of current layer, however A[i-2] denotes perceptron of previous layer
				# W[i-1] denotes all the synaptic weights between layer i-1 and i.
				# Similarly, B[i-1] is bias associated with current layer
			A[i-1] = self.activation(A[i-1])				# appling activation after successfully calucalting z function for all perceptron in layer
		# A.pop()		# poping out the input layer
		# print(A)
		return A
	
	def __backward_phase(self , A , y) :
		Local_gradient = [a.copy() for a in A]			# cloning B for getting accurate shapes of layers and perceptron

		Local_gradient[-2] = (y - A[-2]) * self.activation_derivative(A[-2])

		for layer_num in range (self.n_layers - 2 , -1 , -1) :
			# 2 (as last two layer are input and output) + 1 (last index is 1 less than len) = 3
			# print(layer_num-1 , Local_gradient[layer_num-1].shape , A[layer_num-1].shape )
			for perceptron_num in range (self.layer_sizes[layer_num]) :
				# print('-->' , perceptron_num)
				Local_gradient[layer_num-1][perceptron_num] = np.dot(self.W[layer_num][perceptron_num] , Local_gradient[layer_num])
				# self.W[layer_num][perceptron_num] is weights between current layer's perceptron and right layer (i.e. next layer towards output layer).
				# Local_gradient[layer_num] is calculated local gradient of right layer (i.e. next layer towards output layer)

				# Local_gradient[layer_num-1][perceptron_num] = mul * self.activation_derivative(A[layer_num-1][perceptron_num])
			Local_gradient[layer_num-1] = Local_gradient[layer_num-1] * self.activation_derivative(A[layer_num-1])

		# print(Local_gradient)
		# return np.array(Local_gradient , dtype=np.float ) * np.array(A , dtype= np.float)
		# return np.array(Local_gradient) * np.array(A)

		# need to return the theta = Local gradient * A



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
		if(self.batch_size > X.shape[0]) :
			print('invalid batch size')
			return

		np.random.seed(12)

		self.W[0][0][0] = 1
		self.W[1][0][0] = 1

		for _ in range(int(np.ceil(self.num_epochs / self.batch_size))) :
			# building the batch
			batch_index = np.random.choice(X.shape[0] , self.batch_size , replace=False)
			for data_point in batch_index :
				A = self.__forward_phase(X[data_point,:])
				self.__backward_phase(A , y[data_point,:])

				# updating the theta and delta W
			# forward phase
			# self.__forward_phase([np.array([1.0,2.0,3.0]) , np.zeros((3,))])
			# self.__forward_phase(np.array([1.0,2.0,3.0]))

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

a = MyNeuralNetwork(3 , [3,2,1] , 'tanh' , 0.01 , 'zero' , 4 ,  1)
X = np.array([np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9]),np.array([10,11,12])])
y = np.array([np.array([1]),np.array([0]),np.array([1]),np.array([0])])
a.fit(X,y)