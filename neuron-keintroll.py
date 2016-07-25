import numpy as np
#import matplotlib.pyplot as plt
import random 


class Perceptron:
	def __init__(self, inpsize, id, activation_function):
		self.weights = [random.uniform(-1, 1) for x in range(inpsize)]
		self.bias = 0.0
		self.id = id
		self.inactive = False
		self.activation_function = activation_function

	def activationfunction_der(self, x):
		#return 1 - np.tanh(x)*np.tanh(x)
		h = 0.000000000001
		return (self.activation_function(x+h) - self.activation_function(x))/h

	def propagate(self, inp):
		self.input = inp

		if self.id == True:
			return inp[0]

		if self.inactive == True:
			return 0

		return self.activation_function(inp) * self.weights[0] + self.bias

	def learn(self, optimal_out, alpha, errorterm = None):

		if errorterm == None:
			errorterm = self.output - optimal_out
			out_der = self.activationfunction_der(np.dot(self.input, self.weights) + self.bias)
			self.delta = out_der * errorterm
		else:
			self.delta = errorterm

		self.weights = self.weights + alpha * self.delta*np.transpose(self.input)
		self.bias=self.bias + alpha * self.delta


	#np.tanh


lin_cell = Perceptron(1, False, lambda x : x )
quad_cell = Perceptron(1, False, lambda x : x*x )




def linear_training_data(m, b):
	for x in range (100):
		yield (x/100.0, m*x/100.0 + b)


def quad_training_data(a, b, c):
	for x in range (100):
		yield (x, a*x*x*1.0 + b*x*1.0 + c*1.0)


data = list(quad_training_data(2, 0.0, 0))


for bla in range(100):
	for x,y in data:
		#lin_cell.output = lin_cell.propagate([x])
		quad_cell.output = quad_cell.propagate(x)

		output = quad_cell.output #lin_cell.output + quad_cell.output 
		print x, y, output

		#lin_cell.learn(output, 0.2, y - output)
		quad_cell.learn(output, 0.000002, y - output)



print quad_cell.weights
print quad_cell.bias

#print abs_cell.weights



#plt.scatter([1],[1])
#plot.show()