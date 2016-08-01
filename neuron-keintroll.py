import numpy as np
#import matplotlib.pyplot as plt
import random 



def setup_weights(layer_number, layer_size):
	weights = np.random.rand(layer_number, layer_size)
	return weights

def apply_activation_fns(activation_function, input_data, weights):
	return activation_function(np.dot(input_data, weights))

def delta_fn(derivative_activation_fn, input_data, weights, errors):
	return derivative_activation_fn(np.dot(input_data, weights)) * errors

def error_fn(actual_value, target_value):
	return actual_value - target_value

def error_fn2(weights, deltas):
	return np.dot(weights, deltas)

def learn_fn(derivative_activation_fn, deltas, input_data, weights):
	alpha = 0.2
	#return np.outer(deltas, input_data)
	return weights - alpha * np.outer(input_data, deltas)


def sech(x):
	return 1/np.cosh(x)**2



def train_feed_forward(layer_list, training_data):
	act_fn = np.tanh
	act_fn_derived = sech

	weight_list = []

	for i in range(len(layer_list)-1):
		weight_list.append(setup_weights(layer_list[i], layer_list[i+1]))


	for i in range(len(training_data)):

		input_data = training_data[i,0]
		target_data = training_data[i,1]

		results = []
		results.append(input_data)

		reductor = lambda x, y : x.append(apply_activation_fns(act_fn, x[len(x)-1], y))		

		#reduce(reductor, weight_list, results)

		#feed input forward
		for j in range(len(weight_list)):
			reductor(results, weight_list[j])

		#propagate error backwards
		for j in reversed(range(len(weight_list))):
			if(j == len(weight_list) - 1):
				error_term = error_fn(results[j+1], target_data)
			else:
				error_term = error_fn2(weight_list[j+1], last_deltas)				

			last_deltas = delta_fn(sech, results[j], weight_list[j], error_term)
			weight_list[j] = learn_fn(act_fn_derived, last_deltas, results[j], weight_list[j])
			
	# return the trained weights
	return weight_list


def apply_network(weights, input):
	act_fn = np.tanh
	act_fn_derived = sech

	results = []
	results.append(input)

	reductor = lambda x, y : x.append(apply_activation_fns(act_fn, x[len(x)-1], y))		

	#reduce(reductor, weight_list, results)

	#feed input forward
	for j in range(len(weights)):
		reductor(results, weights[j])

	return results[len(results)-1]





def absolute_training_data():
	training_data = np.tile(np.array([[1,2,3,4], 0.5]), (100, 1) )

	# add random noise
	for i in range(len(training_data)):
		training_data[i, 0] = training_data[i,0] + 0.2 * np.random.rand(4)

	return training_data



# sinusoidal training data
sin_data = np.array([[[x], np.sin(x)] for x in np.arange(-3.14, 3.14, 0.1)])


training_data = sin_data
#training_data = absolute_training_data()

#print training_data



network = train_feed_forward([1,10,10,1], training_data)
print network

result = apply_network(network, 0)


print "result", result



'''
def linear_training_data(m, b):
	for x in range (100):
		yield (x, m*x + b)

data = list(linear_training_data(-1,1))
'''


