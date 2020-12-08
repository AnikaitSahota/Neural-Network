import numpy as np
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import Q1
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers

class Q2() :
	def __init__(self) :
		self.__extract_MNISTdata(0.15)

	def __extract_MNISTdata(self , ratio) :
		# the data, split between train and test sets
		(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

		# Scale images to the [0, 1] range
		x_train = x_train.astype("float32") / 255
		x_test = x_test.astype("float32") / 255

		# making shape from (num_of_datapoints , 28, 28) to (num_of_datapoints , 28 * 28)
		x_train = x_train.reshape(x_train.shape[0] ,x_train.shape[1]*x_train.shape[2] )
		x_test = x_test.reshape(x_test.shape[0] ,x_test.shape[1]*x_test.shape[2] )

		X_bin, self.X_train, y_bin, self.y_train = train_test_split(x_train, y_train, test_size = ratio, random_state = 42, stratify = y_train)
		X_bin, self.X_test, y_bin, self.y_test = train_test_split(x_test, y_test, test_size = ratio, random_state = 42, stratify = y_test)
		# end = int(ratio * x_train.shape[0])
		# self.X_train = x_train[:end]
		# self.y_train = y_train[:end]
		# end = int(1 * x_test.shape[0])
		# self.X_test = x_test[:end]
		# self.y_test = y_test[:end]

	def __plot_errorVSepochs(self , train_error , validation_error , title):
		epochs = [i for i in range(train_error.shape[0])]
		plt.figure()
		plt.plot(epochs , validation_error , 'r-')
		plt.plot(epochs , train_error , 'b-')
		plt.legend(['Validation error' , 'Training error'] , loc = 'upper right')
		plt.xlabel('number of epochs')
		plt.ylabel('loss')
		plt.title(title)

		# plt.show()
		plt.savefig('plot_images/'+title+'.png')
	
	def __plot_cluster(self, X , y , title) :
		"""function to plot the cluster with diffren colors associated with labels

		Args:
			X (numpy 2D array): It is the set of independent variable
			y (numpy 1D aray): It is the dependent vaibale respective to X
		"""
		D1 = X[:,0]								# extracting the dimension one
		D2 = X[:,1]								# extracting the dimension two
		# print(D1.shape , D2.shape, y.shape)
		plt.figure()
		plt.xlabel('D0')
		plt.ylabel('D1')
		a = plt.scatter(D1 , D2 , c = y)			# ploting the scatter plot
		plt.legend(*a.legend_elements(),loc = 'best')
		plt.title(title)
		# plt.show()													# showing the plot
		plt.savefig('plot_images/'+title+'.png')

	def solver(self , load_from_file = True) :
		for acti_fn in ['relu', 'sigmoid', 'linear', 'tanh'] :
			if(load_from_file == False) :
				model = Q1.MyNeuralNetwork(5 , [784,256,128,64,10] , acti_fn , 0.1 , 'normal' , 500 ,  100)
				model.fill_testing_data(self.X_test , self.y_test,verbose=1)
				print('-------'*8)
				print('Begening fit for activation function',acti_fn)
				model.fit(self.X_train, self.y_train)
				joblib.dump(model , 'models/saved_model_'+acti_fn)
			
			else :
				model = joblib.load('models/saved_model_'+acti_fn)
			
			err = model.get_errors_featres()
			self.__plot_errorVSepochs(err[0] , err[1] , 'Actvation function = ' + acti_fn)

			NNfeatures = TSNE(n_components = 2).fit_transform(err[2])
			self.__plot_cluster(NNfeatures , self.y_test , 'Features using '+acti_fn)

			print('Accuracy for',acti_fn,'= ', model.score(self.X_test, self.y_test) * 100 , '%')

	def last_part(self) :
		for acti_fn in ['relu', 'logistic', 'identity', 'tanh'] :
			print('-------'*8)
			print('Begening fit for activation function',acti_fn)
			clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation = acti_fn, learning_rate_init=0.1, batch_size=100, max_iter=100, random_state=7).fit(self.X_train, self.y_train)
			print('Accuracy for',acti_fn,'= ', clf.score(self.X_test, self.y_test) * 100 , '%')



Q2().solver(False)

Q2().last_part()