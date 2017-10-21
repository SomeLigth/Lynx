#!/usr/bin/env python

# Importing random module
import random

# Defining the perceptron
class Perceptron:
	
	# Defining the constructor of the class, if you do not know what is it, just ask me
	def __init__(self, numberOfInputs, stepSize = 0.1):
		# The number of inputs that the perceptron will have
		self.numberOfInputs = numberOfInputs
		# The weight, really, one weight for each input, with the random assignment.
		self.aWeight = [random.random() for _ in range(numberOfInputs)]
		# The step size...
		self.eta = stepSize

	# This function will modify the patterns of the neuron (perceptron)
	def activationFunction(self, inputs):
		# The weighted average is the sum up between all inputs multiplied by their weights
		weightedAverage = sum(oneWeight*inputs for oneWeight,inputs in zip(self.aWeight,inputs))
		if weightedAverage > 0: 
			return 1
		else:
			return 0

	# The train function, this function help you to train the neuron (perceptron), you need give it an
	# array of known elements, these elements are the input, and for each input, you put the expected
	# output, with this, the neuron do an auto-calibration.
	def trainFunction(self, inputs, expectedOutput):
		# Defining the output, it is the return of the activation function
		output = self.activationFunction(inputs)
		# Defining the error, this variable can be used to calibrate the neuron, this mean, the error
		# is the diference between the expected output and the true output
		error = expectedOutput - output
		# Is the process have a error (not a tecnical error, a calc error)
		if error != 0:
			# Then the neuron modify their weight
			self.aWeight = [weight + self.eta*error*x for w,x in zip(self.aWeight,inputs)]
		return error
