import numpy as np
import matplotlib.pyplot as plt
import random 



def setup_weights(layer_height, layer_size):
	weights = np.random.rand(layer_height, layer_size)
	weights = (weights - 0.5)/10
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
	alpha = 0.05
	#return np.outer(deltas, input_data)
	return weights - alpha * np.outer(input_data, deltas)

def sech(x):
	return 1/np.cosh(x)**2


def init_feed_forward(layer_list):
	weight_list = []

	for i in range(len(layer_list)-1):
		weight_list.append(setup_weights(layer_list[i], layer_list[i+1]))

	return weight_list

	
def train_feed_forward(weight_list, training_data):
	act_fn = np.tanh
	act_fn_derived = sech

	

	act_fn = lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda x: np.exp(x) - 1, lambda x: x])
	act_fn_derived = lambda x: np.piecewise(x, [x < 0, x >= 0], [lambda x: np.exp(x), lambda x: 1])

	

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



def normalize_data(data):
	x_max = 0

	for d in data:
		if(abs(d[0][0]) > x_max):
			x_max = abs(d[0][0])

	y_max = 0

	for d in data:
		if(abs(d[1]) > y_max):
			y_max = abs(d[1])

	for i in range(len(data)):
		data[i] = [data[i][0] / x_max, data[i][1] / y_max]

	return [x_max, y_max, data]


def renormalize_data(data, factors):
	for i in range(len(data)):
		data[i] = [data[i][0] * factors[0], data[i][1] * factors[1]]

	return data



# sinusoidal training data
rangex1 = np.arange(-4, 4, 0.1)
rangex = random.sample(rangex1, len(rangex1))
sin_data = np.array([[[x], np.sin(x)] for x in rangex ])
	

training_data = normalize_data(sin_data)

stretch_factors = [training_data[0], training_data[1]]

sin_data = renormalize_data(sin_data, stretch_factors)

print stretch_factors

training_data = training_data[2]

print training_data



start_weights = init_feed_forward([1,80,1])
network = train_feed_forward(start_weights, training_data)

for x in range(1000):
	#training_data2 = random.sample(training_data)
	network = train_feed_forward(network, training_data)

#print network


result = [apply_network(network, x)[0,0] for x in rangex1 ]

#print "result", [rangex, result]


xy_data = zip(rangex1, result)

stretch_factors[0] = 1
xy_data = renormalize_data(xy_data, stretch_factors)

#print xy_data

for x in xy_data:
	plt.scatter(x[0], x[1])

for x in sin_data:
	plt.scatter(x[0], x[1], color='red')

#plt.plot(xy_data[0], xy_data[1])
plt.show()


'''
def linear_training_data(m, b):
	for x in range (100):
		yield (x, m*x + b)

data = list(linear_training_data(-1,1))
'''



def absolute_training_data():
	training_data = np.tile(np.array([[1,2,3,4], 0.5]), (100, 1) )

	# add random noise
	for i in range(len(training_data)):
		training_data[i, 0] = training_data[i,0] + 0.2 * np.random.rand(4)

	return training_data